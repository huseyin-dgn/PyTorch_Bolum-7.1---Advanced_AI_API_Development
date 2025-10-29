# api_server_cnn.py
import socket, threading, json, os, mimetypes, time, base64, io
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from cnn_model import SmallCNN

# ========================= Genel Ayarlar =========================
HOST = "127.0.0.1"
PORT = 8096
BUF = 4096
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))  # text.html/style.js aynı klasör
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")
mimetypes.init()

RESET, RED, GREEN, YELLOW, BLUE, CYAN = "\033[0m","\033[91m","\033[92m","\033[93m","\033[94m","\033[96m"
def log(msg, color=BLUE, symbol="💬"):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} {msg}"
    print(f"{color}{line}{RESET}")
    with open(LOG_FILE,"a",encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

# ========================= Route Sistemi =========================
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

# ========================= CNN Yükleme =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_WEIGHTS = os.path.join(STATIC_DIR, "cnn.pt")
LABELS_PATH   = os.path.join(STATIC_DIR, "labels.json")

_model = None
_labels = None

def _ensure_labels():
    global _labels
    if _labels is None:
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError("labels.json bulunamadı")
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _labels = json.load(f)
    return _labels

def load_cnn():
    global _model
    _ensure_labels()
    num_classes = len(_labels)
    m = SmallCNN(num_classes=num_classes).to(DEVICE)
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError("cnn.pt bulunamadı")
    sd = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m
    log(f"CNN yüklendi (classes={num_classes}, device={DEVICE})", GREEN, "🧠")

# ========================= Görüntü İşleme =========================
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img: Image.Image):
    # 224x224, RGB, normalize -> torch [1,3,224,224]
    img = img.convert("RGB").resize((224,224), Image.BICUBIC)
    arr = np.array(img).astype(np.float32)/255.0
    arr = (arr - MEAN)/STD
    arr = np.transpose(arr, (2,0,1))  # HWC->CHW
    t = torch.from_numpy(arr).unsqueeze(0)  # [1,3,224,224]
    return t.to(DEVICE)

def decode_b64_to_image(b64str: str) -> Image.Image:
    # "data:image/xxx;base64,...." ise header’ı at.
    if "," in b64str and b64str.lstrip().startswith("data:"):
        b64str = b64str.split(",",1)[1]
    raw = base64.b64decode(b64str)
    return Image.open(io.BytesIO(raw))

# ========================= CORS Preflight =========================
@route("OPTIONS", "/predict")
def cors_preflight_predict(headers=None, body=None):
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

# ========================= Endpointler =========================
@route("GET", "/")
def root():
    return "<h1>🧠 CNN API</h1><p>Frontend → /text.html • POST /predict (JSON base64)</p>"

@route("GET", "/health")
def health():
    ok = _model is not None
    return {"ok": ok, "device": str(DEVICE)}

@route("POST", "/reload")
def reload_model(headers, body):
    load_cnn()
    return {"status":"ok","message":"CNN yeniden yüklendi","classes":len(_labels)}

@route("POST", "/predict")
def predict(headers, body):
    if _model is None:
        load_cnn()
    data = json.loads(body or "{}")
    image_b64 = data.get("image_b64","").strip()
    if not image_b64:
        return (400, "application/json", json.dumps({"error":"image_b64 eksik"}, ensure_ascii=False))

    img = decode_b64_to_image(image_b64)
    tensor = preprocess_image(img)  # [1,3,224,224]
    with torch.no_grad():
        logits = _model(tensor)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy().tolist()

    labels = _ensure_labels()
    # top5
    topk = sorted([(i,p) for i,p in enumerate(probs)], key=lambda x: x[1], reverse=True)[:5]
    result = [{"label": labels[i] if i < len(labels) else str(i), "prob": float(p)} for i,p in topk]
    return {"topk": result}

# ========================= Statik Dosya =========================
def serve_static(path):
    if path in ("","/"): path="text.html"
    full=os.path.join(STATIC_DIR,path.lstrip("/"))
    if not os.path.exists(full): raise FileNotFoundError(f"{path} bulunamadı.")
    ctype,_=mimetypes.guess_type(full)
    if not ctype: ctype="application/octet-stream"
    with open(full,"rb") as f: body=f.read()
    return 200,ctype,body

# ========================= Request/Client =========================
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
            if method.upper()=="POST": raw=handler(headers,body)
            else: raw=handler()
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
        conn.sendall(build_http_response(status,ctype,body,extra))
        conn.close()
        color=GREEN if status<400 else (YELLOW if status<500 else RED)
        log(f"{status} → {path} ⏱ {elapsed:.2f}ms",color,"✅" if status<400 else "⚠️")

# ========================= Boot =========================
if __name__ == "__main__":
    # İlk açılışta yüklemeye çalış (weights yoksa /reload çağırabilirsin)
    try:
        load_cnn()
    except Exception as e:
        log(f"Model yüklenemedi (devam ediyorum): {e}", YELLOW, "⚠️")

    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        srv.bind((HOST,PORT))
        srv.listen(64)
        log(f"🚀 CNN API çalışıyor → http://{HOST}:{PORT}", CYAN)
        while True:
            conn,addr=srv.accept()
            threading.Thread(target=handle_client,args=(conn,addr)).start()
