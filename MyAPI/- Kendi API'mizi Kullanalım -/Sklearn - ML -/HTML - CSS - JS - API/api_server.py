# ==========================================================
# 🧩 AŞAMA 10.1 — CORS (Tarayıcı Erişimi Aktif)
# ==========================================================
import socket, threading, json, os, mimetypes, time
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

HOST = "127.0.0.1"
PORT = 8097
BUF = 4096
STATIC_DIR = r"C:\Users\hdgn5\OneDrive\Masaüstü\Kendi API'mız\- Kendi API'mizi Kullanalım -\Sklearn - ML -\HTML - CSS - JS - API"

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")
mimetypes.init()

# ==========================================================
# LOG SİSTEMİ
# ==========================================================
RESET, RED, GREEN, YELLOW, BLUE, CYAN = "\033[0m","\033[91m","\033[92m","\033[93m","\033[94m","\033[96m"

def log(msg, color=BLUE, symbol="💬"):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} {msg}"
    print(f"{color}{line}{RESET}")
    with open(LOG_FILE,"a",encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

# ==========================================================
# ROUTE SİSTEMİ
# ==========================================================
ROUTES: Dict[Tuple[str,str], callable] = {}

def route(method, path):
    def decorator(func):
        ROUTES[(method.upper(), path)] = func
        log(f"Route eklendi → {method.upper()} {path}", CYAN, "📍")
        return func
    return decorator

# ==========================================================
# AUTO RESPONSE + HEADER
# ==========================================================
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
    # 🔥 GLOBAL CORS HEADER
    extra_headers.setdefault("Access-Control-Allow-Origin","*")

    header_lines = [
        f"HTTP/1.1 {status} {status_text}",
        f"Content-Type: {ctype}; charset=utf-8",
        f"Content-Length: {len(body_bytes)}",
        "Connection: close",
    ] + [f"{k}: {v}" for k,v in extra_headers.items()]

    return ("\r\n".join(header_lines) + "\r\n\r\n").encode("ascii") + body_bytes

# ==========================================================
# CORS PRE-FLIGHT (OPTIONS /predict)
# ==========================================================
@route("OPTIONS", "/predict")
def cors_preflight(headers=None, body=None):
    return (
        200,
        "text/plain",
        "",
        {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )

# ==========================================================
# ENDPOINTLER
# ==========================================================
@route("GET", "/")
def root():
    return "<h1>🧠 Sklearn Model API</h1><p>Frontend → /text.html</p>"

@route("POST", "/predict")
def predict(headers, body):
    import pickle, numpy as np
    data = json.loads(body)
    X = data.get("x", [])

    if not X:
        raise ValueError("JSON içinde 'x' değeri eksik")

    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("model.pkl bulunamadı")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)  # 🔥 1 satır, n sütun
    preds = model.predict(X).tolist()

    return {"prediction": preds, "count": len(preds)}


# ==========================================================
# STATİK DOSYA SERVİSİ
# ==========================================================
def serve_static(path):
    if path in ("","/"): path="text.html"
    full=os.path.join(STATIC_DIR,path.lstrip("/"))
    if not os.path.exists(full): raise FileNotFoundError(f"{path} bulunamadı.")
    ctype,_=mimetypes.guess_type(full)
    if not ctype: ctype="application/octet-stream"
    with open(full,"rb") as f: body=f.read()
    return 200,ctype,body

# ==========================================================
# REQUEST PARSING
# ==========================================================
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

# ==========================================================
# CLIENT HANDLER
# ==========================================================
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

# ==========================================================
# SUNUCU
# ==========================================================
with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    srv.bind((HOST,PORT))
    srv.listen(64)
    log(f"🚀 CORS destekli API çalışıyor → http://{HOST}:{PORT}",CYAN)
    while True:
        conn,addr=srv.accept()
        threading.Thread(target=handle_client,args=(conn,addr)).start()
