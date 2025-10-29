# ==========================================================
# 🧠 Minimal LLM API (Socket tabanlı)
# - Dinamik route dekoratörü (GET/POST/OPTIONS)
# - Statik dosya servisi (text.html / style.css / script.js)
# - CORS (tarayıcıdan fetch serbest)
# - Vocab + Model yükleme (eğitim mimarisiyle birebir)
# - /generate (metin üretimi) & /reload (modeli yeniden yükle)
# - X-Response-Time header
# ==========================================================
import socket, threading, json, os, mimetypes, time, math, warnings
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Genel Ayarlar
# ---------------------------
HOST = "127.0.0.1"
PORT = 8097
BUF = 4096

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = SCRIPT_DIR  # aynı klasörden text.html, style.css, script.js
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

mimetypes.init()
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Renkli Log
# ---------------------------
RESET, RED, GREEN, YELLOW, BLUE, CYAN = "\033[0m","\033[91m","\033[92m","\033[93m","\033[94m","\033[96m"
def log(msg, color=BLUE, symbol="💬"):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} {msg}"
    print(f"{color}{line}{RESET}")
    with open(LOG_FILE,"a",encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

# ==========================================================
# --------------------------- LLM Bileşenleri --------------
# ==========================================================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLM_WEIGHTS  = os.path.join(SCRIPT_DIR, "llm.pt")
LLM_VOCAB    = os.path.join(SCRIPT_DIR, "vocab.json")

# 🔧 Eğitimle birebir aynı hiperparametreleri gir:
EMBED_DIM    = 256
NUM_HEADS    = 4
LAYERS_ENC   = 4
LAYERS_DEC   = 4
MAX_LEN      = 128
EXPANSION    = 4
DROPOUT      = 0.1
DROP_PATH    = 0.1
USE_SWIGLU   = False

# ---- Token/Mask yardımcıları
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
stoi: Dict[str,int] = {}
itos: list[str] = []

def load_vocab(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"vocab.json bulunamadı: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"vocab.json boş dosya: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError(f"vocab.json içerik boş: {path}")
    try:
        vocab = json.loads(raw)
    except json.JSONDecodeError as e:
        snippet = raw[:120].replace("\n","\\n")
        raise ValueError(f"vocab.json JSON bozuk: {e} | İlk karakterler: {snippet}")
    if "stoi" in vocab and "itos" in vocab:
        _stoi = vocab["stoi"]
        _itos = vocab["itos"]
    else:
        _stoi = vocab
        _itos = [None] * (max(_stoi.values()) + 1)
        for k, v in _stoi.items():
            _itos[v] = k
    for must in (PAD, BOS, EOS, UNK):
        if must not in _stoi:
            raise ValueError(f"Vocab içinde zorunlu token eksik: {must}")
    return _stoi, _itos

# ---- Model katmanları
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.2): 
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training: return x
        keep = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rnd.floor_()
        return x.div(keep) * rnd

class TokenEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim): 
        super().__init__(); self.embedding = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x): return self.embedding(x)

class PositionelEncod(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0)/embed_dim))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=16, dp=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dp)
    def split(self, x):
        B, T, C = x.size()
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
    def merge(self, x):
        B, H, T, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H*D)
    def forward(self, q, k, v, mask=None):
        Q = self.split(self.q_proj(q)); K = self.split(self.k_proj(k)); V = self.split(self.v_proj(v))
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            if mask.dim() == 2:      # [B,T] -> [B,1,1,T]
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:    # [B,T_q,T_k] -> [B,1,T_q,T_k]
                mask = mask[:, None, :, :]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return self.out_proj(self.merge(out))

class FeedForward(nn.Module):
    def __init__(self, embed_dim, expansion=8, dp=0.1, use_swiglu=False):
        super().__init__()
        if use_swiglu:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*expansion*2),
                nn.SiLU(), nn.Dropout(dp),
                nn.Linear(embed_dim*expansion, embed_dim),
                nn.Dropout(dp)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim*expansion),
                nn.GELU(), nn.Dropout(dp),
                nn.Linear(embed_dim*expansion, embed_dim),
                nn.Dropout(dp)
            )
    def forward(self, x): return self.net(x)

class TransformerEncoderBlockLLM(nn.Module):
    def __init__(self, embed_dim, num_heads, dp, drop_path, expansion, use_swiglu):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim); self.norm2 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dp)
        self.ffn = FeedForward(embed_dim, expansion, dp, use_swiglu)
        self.drop_path = DropPath(drop_path)
        self.gamma_1 = nn.Parameter(torch.ones(embed_dim)*1e-2)
        self.gamma_2 = nn.Parameter(torch.ones(embed_dim)*1e-2)
    def forward(self, x, mask=None):
        a = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.drop_path(self.gamma_1 * a)
        f = self.ffn(self.norm2(x))
        x = x + self.drop_path(self.gamma_2 * f)
        return x

class TransformersEncoderLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, dp, num_heads, expansion, max_len, drop_path, use_swiglu):
        super().__init__()
        self.tok_emb = TokenEmbed(vocab_size, embed_dim)
        self.pos_enc = PositionelEncod(embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlockLLM(embed_dim, num_heads, dp, drop_path, expansion, use_swiglu)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, src_tokens, src_mask=None):
        x = self.tok_emb(src_tokens); x = self.pos_enc(x)
        for lyr in self.layers:
            x = lyr(x, mask=src_mask)
        return self.norm(x)

class TransformerDecoderBlockLLM(nn.Module):
    def __init__(self, embed_dim, num_heads, dp, drop_path, expansion, use_swiglu):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim); self.norm2 = nn.LayerNorm(embed_dim); self.norm3 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dp)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dp)
        self.ffn = FeedForward(embed_dim, expansion, dp, use_swiglu)
        self.drop_path = DropPath(drop_path)
        self.gamma_1 = nn.Parameter(torch.ones(embed_dim)*1e-2)
        self.gamma_2 = nn.Parameter(torch.ones(embed_dim)*1e-2)
        self.gamma_3 = nn.Parameter(torch.ones(embed_dim)*1e-2)
    def forward(self, x, enc_out=None, self_mask=None, enc_mask=None):
        if self_mask is not None and self_mask.dim() == 2:
            causal = torch.tril(torch.ones((x.size(1), x.size(1)), device=x.device)).bool()
            self_mask = self_mask[:, None, :] & causal[None, :, :]
        a = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), self_mask)
        x = x + self.drop_path(self.gamma_1 * a)
        if enc_out is not None:
            if enc_mask is not None and enc_mask.dim() == 2:
                enc_mask = enc_mask[:, None, None, :]
            c = self.cross_attn(self.norm2(x), self.norm2(enc_out), self.norm2(enc_out), enc_mask)
            x = x + self.drop_path(self.gamma_2 * c)
        f = self.ffn(self.norm3(x))
        x = x + self.drop_path(self.gamma_3 * f)
        return x

class TransformerDecoderLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dp, drop_path, expansion, max_len, use_swiglu):
        super().__init__()
        self.embedding = TokenEmbed(vocab_size, embed_dim)
        self.pos_encoding = PositionelEncod(embed_dim, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderBlockLLM(embed_dim, num_heads, dp, drop_path, expansion, use_swiglu)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
    def forward(self, x, enc_out=None, self_mask=None, enc_mask=None):
        x = self.embedding(x); x = self.pos_encoding(x)
        for lyr in self.layers:
            x = lyr(x, enc_out, self_mask, enc_mask)
        x = self.norm(x)
        return self.lm_head(x)

class Seq2SeqLLM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder; self.decoder = decoder
    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src_tokens, src_mask)
        return self.decoder(tgt_tokens, enc_out, self_mask=tgt_mask, enc_mask=src_mask)

def build_model(vocab_size:int,
                embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                num_layers_enc=LAYERS_ENC, num_layers_dec=LAYERS_DEC,
                dp=DROPOUT, drop_path=DROP_PATH, expansion=EXPANSION,
                max_len=MAX_LEN, use_swiglu=USE_SWIGLU):
    enc = TransformersEncoderLLM(
        vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers_enc,
        dp=dp, num_heads=num_heads, expansion=expansion, max_len=max_len,
        drop_path=drop_path, use_swiglu=use_swiglu
    )
    dec = TransformerDecoderLLM(
        vocab_size=vocab_size, embed_dim=embed_dim, num_layers=num_layers_dec,
        num_heads=num_heads, dp=dp, drop_path=drop_path, expansion=expansion,
        max_len=max_len, use_swiglu=use_swiglu
    )
    return Seq2SeqLLM(enc, dec)

def encode(text: str, max_len=256):
    toks = text.strip().split()
    ids = [stoi.get(BOS, 1)] + [stoi.get(t, stoi.get(UNK, 3)) for t in toks] + [stoi.get(EOS, 2)]
    ids = ids[:max_len]
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)

def decode(ids):
    toks = []
    for i in ids:
        tok = itos[int(i)]
        if tok == EOS: break
        if tok in (BOS, PAD): continue
        toks.append(tok)
    return " ".join(toks)

_llm: Optional[Seq2SeqLLM] = None

def load_llm():
    """Eğitimle aynı mimari + weight tying + state dict yükleme"""
    global _llm, stoi, itos
    if not os.path.exists(LLM_VOCAB):
        raise FileNotFoundError(f"vocab.json bulunamadı: {LLM_VOCAB}")
    if not os.path.exists(LLM_WEIGHTS):
        raise FileNotFoundError(f"llm.pt bulunamadı: {LLM_WEIGHTS}")

    stoi, itos = load_vocab(LLM_VOCAB)
    vocab_size = len(itos)

    _llm = build_model(vocab_size=vocab_size).to(DEVICE)

    # weight tying (eğitimde yaptıysan burada da yap)
    _llm.decoder.lm_head.weight = _llm.decoder.embedding.embedding.weight

    sd = torch.load(LLM_WEIGHTS, map_location=DEVICE)
    _llm.load_state_dict(sd, strict=True)
    _llm.eval()
    log(f"🧠 LLM yüklendi | vocab={vocab_size}, device={DEVICE}, embed={EMBED_DIM}, heads={NUM_HEADS}, "
        f"enc_layers={LAYERS_ENC}, dec_layers={LAYERS_DEC}, max_len={MAX_LEN}", GREEN, "🧠")

@torch.no_grad()
def generate(prompt: str, max_new_tokens=64, temperature=1.0):
    """Greedy/top-k yok; temperature>0 ise sampling, 0 ise argmax."""
    _llm.eval()
    src = encode(prompt, max_len=MAX_LEN)
    src_mask = (src != stoi.get(PAD, 0)).to(src.device)
    tgt = torch.tensor([[stoi.get(BOS,1)]], dtype=torch.long, device=DEVICE)
    enc_out = _llm.encoder(src, src_mask)

    for _ in range(max_new_tokens):
        self_mask = torch.ones((1, tgt.size(1)), dtype=torch.bool, device=DEVICE)
        logits = _llm.decoder(tgt, enc_out, self_mask, src_mask)[:, -1, :]
        if temperature and temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_id], dim=1)
        if int(next_id.item()) == stoi.get(EOS, 2):
            break

    out = tgt[0].tolist()
    return decode(out), out

# ==========================================================
# --------------------------- HTTP Katmanı -----------------
# ==========================================================
ROUTES: Dict[Tuple[str,str], callable] = {}
def route(method, path):
    def decorator(func):
        ROUTES[(method.upper(), path)] = func
        log(f"Route eklendi → {method.upper()} {path}", CYAN, "📍")
        return func
    return decorator

def make_response_auto(data: Any):
    if isinstance(data, tuple): return data
    if isinstance(data, dict):  return 200, "application/json", json.dumps(data, ensure_ascii=False)
    if isinstance(data, str):   return 200, "text/html", data
    if isinstance(data,(bytes,bytearray)): return 200,"application/octet-stream",data
    return 500, "application/json", json.dumps({"error": str(type(data))}, ensure_ascii=False)

def build_http_response(status, ctype, body, extra_headers=None):
    if isinstance(body,str): body_bytes = body.encode("utf-8")
    elif isinstance(body,(bytes,bytearray)): body_bytes = bytes(body)
    else:
        body_bytes = json.dumps(body,ensure_ascii=False).encode("utf-8")
        ctype="application/json"
    status_text = {200:"OK",400:"Bad Request",404:"Not Found",500:"Internal Server Error"}.get(status,"OK")
    if extra_headers is None: extra_headers={}
    # Global CORS
    extra_headers.setdefault("Access-Control-Allow-Origin","*")
    header_lines = [
        f"HTTP/1.1 {status} {status_text}",
        f"Content-Type: {ctype}; charset=utf-8",
        f"Content-Length: {len(body_bytes)}",
        "Connection: close",
    ] + [f"{k}: {v}" for k,v in extra_headers.items()]
    return ("\r\n".join(header_lines) + "\r\n\r\n").encode("ascii") + body_bytes

def serve_static(path):
    if path in ("","/"): path="text.html"
    full=os.path.join(STATIC_DIR,path.lstrip("/"))
    if not os.path.exists(full): raise FileNotFoundError(f"{path} bulunamadı.")
    ctype,_=mimetypes.guess_type(full)
    if not ctype: ctype="application/octet-stream"
    with open(full,"rb") as f: body=f.read()
    return 200,ctype,body

def parse_request(conn):
    data=b""
    while b"\r\n\r\n" not in data:
        chunk=conn.recv(BUF)
        if not chunk: break
        data+=chunk
    headers_part,_,body_bytes=data.partition(b"\r\n\r\n")
    lines=headers_part.decode("utf-8",errors="ignore").split("\r\n")
    if not lines or not lines[0]: return "GET","/",{}, ""
    method,path,_=lines[0].split()
    headers={}
    for line in lines[1:]:
        if ":" in line:
            k,v=line.split(":",1)
            headers[k.strip().lower()]=v.strip()
    content_length=int(headers.get("content-length","0"))
    while len(body_bytes)<content_length:
        body_bytes+=conn.recv(BUF)
    body=body_bytes.decode("utf-8",errors="ignore")
    return method,path.rstrip("/") or "/",headers,body

def handle_client(conn,addr):
    start=time.perf_counter()
    try:
        method,path,headers,body=parse_request(conn)
        log(f"🔎 İstek → ({method}, {path})", YELLOW)
        key=(method.upper(),path)
        if key in ROUTES:
            handler=ROUTES[key]
            raw=handler(headers,body) if method.upper()=="POST" else handler()
            status,ctype,body=make_response_auto(raw)
        else:
            raw=serve_static(path)
            status,ctype,body=make_response_auto(raw)
    except Exception as e:
        status,ctype,body=500,"application/json",json.dumps({"error":str(e)},ensure_ascii=False)
        log(f"❌ Hata: {e}",RED)
    finally:
        elapsed=(time.perf_counter()-start)*1000
        extra={"X-Response-Time":f"{elapsed:.2f}ms"}
        try:
            conn.sendall(build_http_response(status,ctype,body,extra))
        except Exception as e:
            log(f"Yanıt gönderilemedi: {e}", RED)
        conn.close()
        color=GREEN if status<400 else (YELLOW if status<500 else RED)
        log(f"{status} → {path} ⏱ {elapsed:.2f}ms",color,"✅" if status<400 else "⚠️")

# ==========================================================
# --------------------------- Endpoint'ler -----------------
# ==========================================================
@route("OPTIONS", "/generate")
def cors_preflight_generate(headers=None, body=None):
    return (200,"text/plain","",{
        "Access-Control-Allow-Origin":"*",
        "Access-Control-Allow-Methods":"POST, OPTIONS",
        "Access-Control-Allow-Headers":"Content-Type",
    })

@route("OPTIONS", "/reload")
def cors_preflight_reload(headers=None, body=None):
    return (200,"text/plain","",{
        "Access-Control-Allow-Origin":"*",
        "Access-Control-Allow-Methods":"POST, OPTIONS",
        "Access-Control-Allow-Headers":"Content-Type",
    })

@route("GET", "/")
def root():
    return "<h1>🧠 LLM API</h1><p>POST /generate ile metin üret. Frontend → /text.html</p>"

@route("GET", "/health")
def health():
    ok = _llm is not None
    return {"ok": ok, "device": str(DEVICE)}

@route("POST", "/reload")
def reload_llm(headers, body):
    load_llm()
    return {"status":"ok","message":"LLM yeniden yüklendi"}

@route("POST", "/generate")
def generate_text(headers, body):
    if _llm is None:
        return (500,"application/json",json.dumps({"error":"Model yüklenmedi"},ensure_ascii=False))
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return (400,"application/json",json.dumps({"error":"Geçersiz JSON"},ensure_ascii=False))
    prompt = (data.get("prompt") or "").strip()
    max_new = int(data.get("max_new_tokens", 64))
    temperature = float(data.get("temperature", 1.0))
    if not prompt:
        return (400,"application/json",json.dumps({"error":"prompt eksik"},ensure_ascii=False))
    text, tok_ids = generate(prompt, max_new_tokens=max_new, temperature=temperature)
    return {"response": text, "tokens": tok_ids, "time_ms": None}

# ==========================================================
# --------------------------- BOOT -------------------------
# ==========================================================
if __name__ == "__main__":
    load_llm()  # ilk açılışta model/vocab yükle
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        srv.bind((HOST,PORT))
        srv.listen(64)
        log(f"🚀 LLM API çalışıyor → http://{HOST}:{PORT}",CYAN)
        while True:
            conn,addr=srv.accept()
            threading.Thread(target=handle_client,args=(conn,addr)).start()
