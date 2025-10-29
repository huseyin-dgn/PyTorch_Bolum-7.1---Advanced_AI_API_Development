# ==========================================================
# 🧩 AŞAMA 9.6 — Renkli Log Sistemi + Zaman Damgası
# ==========================================================

import socket
import threading
import json
import os
import mimetypes
from datetime import datetime
from typing import Any

# ---- Genel Ayarlar ----
HOST = "127.0.0.1"
PORT = 8096
BUF = 4096
STATIC_DIR = r"C:\Users\hdgn5\OneDrive\Masaüstü\Kendi API'mız\- API Geliştirne Adımları -\Part 3"

mimetypes.init()

# ==========================================================
# 🎨 1️⃣ Renk Kodları ve Log Fonksiyonu
# ==========================================================
RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"

def log(msg: str, color=BLUE, symbol="💬"):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{now}] {symbol} {msg}{RESET}")

# ==========================================================
# 2️⃣ Dinamik Route Sistemi
# ==========================================================
ROUTES = {}

def route(method, path):
    def decorator(func):
        ROUTES[(method.upper(), path)] = func
        log(f"Route eklendi → {method.upper()} {path} -> {func.__name__}", CYAN, "📍")
        return func
    return decorator

# ==========================================================
# 3️⃣ Auto Response (dict→JSON, str→HTML, bytes→Binary)
# ==========================================================
def make_response_auto(data: Any):
    if isinstance(data, tuple):
        return data
    if isinstance(data, dict):
        return 200, "application/json", json.dumps(data, ensure_ascii=False)
    if isinstance(data, str):
        return 200, "text/html", data
    if isinstance(data, (bytes, bytearray)):
        return 200, "application/octet-stream", bytes(data)
    return 500, "application/json", json.dumps({"error": f"Unsupported type: {type(data)}"}, ensure_ascii=False)

def build_http_response(status: int, ctype: str, body):
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    elif isinstance(body, (bytes, bytearray)):
        body_bytes = bytes(body)
    else:
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        ctype = "application/json"

    status_text = {200:"OK",400:"Bad Request",404:"Not Found",500:"Internal Server Error"}.get(status, "OK")

    headers = (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: {ctype}; charset=utf-8\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii")

    return headers + body_bytes

# ==========================================================
# 4️⃣ Global Error Handler
# ==========================================================
def global_error_handler(func, *args):
    try:
        return func(*args)
    except FileNotFoundError as e:
        log(f"404 Not Found → {e}", YELLOW, "⚠️")
        return (404, "application/json", json.dumps({"error": str(e), "type": "FileNotFoundError"}, ensure_ascii=False))
    except ValueError as e:
        log(f"400 Bad Request → {e}", YELLOW, "⚠️")
        return (400, "application/json", json.dumps({"error": str(e), "type": "ValueError"}, ensure_ascii=False))
    except KeyError as e:
        log(f"Eksik anahtar → {e}", YELLOW, "⚠️")
        return (400, "application/json", json.dumps({"error": f"Eksik anahtar: {e}", "type": "KeyError"}, ensure_ascii=False))
    except Exception as e:
        log(f"500 Internal Error → {type(e).__name__}: {e}", RED, "❌")
        return (500, "application/json", json.dumps({"error": str(e), "type": type(e).__name__}, ensure_ascii=False))

# ==========================================================
# 5️⃣ HTTP Parsing
# ==========================================================
def parse_request(conn):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(BUF)
        if not chunk:
            break
        data += chunk

    headers_part, _, body_bytes = data.partition(b"\r\n\r\n")
    lines = headers_part.decode("utf-8", errors="ignore").split("\r\n")
    method, path, _ = lines[0].split()
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    content_length = int(headers.get("content-length", "0")) if headers else 0
    while len(body_bytes) < content_length:
        body_bytes += conn.recv(BUF)

    body = body_bytes.decode("utf-8", errors="ignore")
    return method, path.rstrip("/") or "/", headers, body

# ==========================================================
# 6️⃣ Endpoint’ler
# ==========================================================
@route("GET", "/")
def root():
    return "<h1>Renkli Log Sistemi ⚙️</h1><p>Artık her şey renklendi!</p>"

@route("POST", "/sum")
def sum_numbers(headers, body):
    data = json.loads(body)
    a, b = float(data["a"]), float(data["b"])
    return {"sum": a + b}

@route("POST", "/sum/plain")
def sum_plain(headers, body):
    data = json.loads(body)
    a, b = float(data["a"]), float(data["b"])
    return (200, "text/plain", str(a + b))

# ==========================================================
# 7️⃣ Statik Dosya Servisi
# ==========================================================
def serve_static(path):
    if path in ("", "/"):
        path = "text.html"
    full_path = os.path.join(STATIC_DIR, path.lstrip("/"))
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"{path} bulunamadı.")
    ctype, _ = mimetypes.guess_type(full_path)
    if not ctype:
        ctype = "application/octet-stream"
    with open(full_path, "rb") as f:
        body = f.read()
    return 200, ctype, body

# ==========================================================
# 8️⃣ İstek İşleyici (Thread başına)
# ==========================================================
def handle_client(conn, addr):
    log(f"Yeni bağlantı: {addr}", BLUE, "📡")
    try:
        method, path, headers, body = parse_request(conn)
        key = (method.upper(), path)
        log(f"İstek geldi → {key}", YELLOW, "🔎")

        if key in ROUTES:
            handler = ROUTES[key]
            raw = global_error_handler(handler, headers, body) if method == "POST" else global_error_handler(handler)
        else:
            raw = global_error_handler(serve_static, path)

        status, ctype, resp_body = make_response_auto(raw)
        log(f"{status} → {path}", GREEN, "✅")
    except Exception as e:
        log(f"Sunucu hatası: {e}", RED, "💥")
        status, ctype, resp_body = 500, "application/json", json.dumps({"error": str(e)}, ensure_ascii=False)

    conn.sendall(build_http_response(status, ctype, resp_body))
    conn.close()
    log(f"Bağlantı kapatıldı: {addr}", CYAN, "🔚")

# ==========================================================
# 9️⃣ Sunucu Başlatma
# ==========================================================
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(64)
    log(f"Renkli Log API çalışıyor → http://{HOST}:{PORT}", CYAN, "🚀")

    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()
