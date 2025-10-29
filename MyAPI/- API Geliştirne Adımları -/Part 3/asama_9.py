# ==========================================================
# 🧩 AŞAMA 9 — Renkli + Dosyalı Log + Süre Ölçümü (X-Response-Time)
# ==========================================================
# Özellikler:
# - Dinamik route dekoratörü
# - Auto Response (dict→JSON, str→HTML, bytes→Binary)
# - Global Error Handler
# - Statik dosya sunumu (binary, doğru MIME)
# - Doğru Content-Length (byte cinsinden)
# - Renkli terminal logları + logs/server.log dosyasına kalıcı kayıt
# - ⏱ İstek süresi ölçümü (ms) + "X-Response-Time" header’ı
# ==========================================================

import socket
import threading
import json
import os
import mimetypes
import time
from datetime import datetime
from typing import Any, Dict, Tuple, Optional

# ---- Genel Ayarlar ----
HOST = "127.0.0.1"
PORT = 8098
BUF = 4096
STATIC_DIR = r"C:\Users\hdgn5\OneDrive\Masaüstü\Kendi API'mız\- API Geliştirne Adımları -\Part 3"

# Log klasörü ve dosyası
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "server.log")

mimetypes.init()

# ==========================================================
# 🎨 Renk Kodları + Log Fonksiyonları
# ==========================================================
RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"

def _file_log_line(msg: str) -> str:
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"

def log_to_file(msg: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(_file_log_line(msg) + "\n")

def log(msg: str, color=BLUE, symbol="💬"):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} {msg}"
    print(f"{color}{line}{RESET}")
    log_to_file(line)

# ==========================================================
# Dinamik Route Sistemi
# ==========================================================
ROUTES: Dict[Tuple[str, str], callable] = {}

def route(method: str, path: str):
    def decorator(func):
        ROUTES[(method.upper(), path)] = func
        log(f"Route eklendi → {method.upper()} {path} -> {func.__name__}", CYAN, "📍")
        return func
    return decorator

# ==========================================================
# Auto Response (dict→JSON, str→HTML, bytes→Binary)
# ==========================================================
def make_response_auto(data: Any) -> Tuple[int, str, Any]:
    if isinstance(data, tuple):
        return data
    if isinstance(data, dict):
        return 200, "application/json", json.dumps(data, ensure_ascii=False)
    if isinstance(data, str):
        return 200, "text/html", data
    if isinstance(data, (bytes, bytearray)):
        return 200, "application/octet-stream", bytes(data)
    return 500, "application/json", json.dumps({"error": f"Unsupported type: {type(data)}"}, ensure_ascii=False)

def build_http_response(status: int, ctype: str, body: Any, extra_headers: Optional[Dict[str, str]]=None) -> bytes:
    # Body → bytes
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    elif isinstance(body, (bytes, bytearray)):
        body_bytes = bytes(body)
    else:
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        ctype = "application/json"

    status_text = {200:"OK",400:"Bad Request",404:"Not Found",500:"Internal Server Error"}.get(status, "OK")

    header_lines = [
        f"HTTP/1.1 {status} {status_text}",
        f"Content-Type: {ctype}; charset=utf-8",
        f"Content-Length: {len(body_bytes)}",
        "Connection: close",
    ]
    if extra_headers:
        for k, v in extra_headers.items():
            header_lines.append(f"{k}: {v}")

    headers = ("\r\n".join(header_lines) + "\r\n\r\n").encode("ascii")
    return headers + body_bytes

# ==========================================================
# Global Error Handler
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
# HTTP Parsing (Content-Length destekli)
# ==========================================================
def parse_request(conn: socket.socket):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(BUF)
        if not chunk:
            break
        data += chunk

    headers_part, _, body_bytes = data.partition(b"\r\n\r\n")
    lines = headers_part.decode("utf-8", errors="ignore").split("\r\n")
    if not lines or lines[0].strip() == "":
        return "GET", "/", {}, ""

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
    path = (path or "/").rstrip("/") or "/"
    return method, path, headers, body

# ==========================================================
# Endpoint’ler
# ==========================================================
@route("GET", "/")
def root():
    return "<h1>⏱ Aşama 9.8</h1><p>Renkli log + dosyaya kayıt + süre ölçümü aktif. (X-Response-Time)</p>"

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
# Statik Dosya Servisi (binary okuma + doğru MIME)
# ==========================================================
def serve_static(path: str):
    if path in ("", "/"):
        path = "text.html"
    full_path = os.path.join(STATIC_DIR, path.lstrip("/"))
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise FileNotFoundError(f"{path} bulunamadı.")
    ctype, _ = mimetypes.guess_type(full_path)
    if not ctype:
        ctype = "application/octet-stream"
    with open(full_path, "rb") as f:
        body = f.read()
    return 200, ctype, body

# ==========================================================
# İstek İşleyici (Thread başına) + Süre Ölçümü
# ==========================================================
def handle_client(conn, addr):
    start = time.perf_counter()
    log(f"Yeni bağlantı: {addr}", BLUE, "📡")

    status, ctype, resp_body = 500, "application/json", json.dumps({"error": "uninitialized"}, ensure_ascii=False)
    extra_headers = {}

    try:
        method, path, headers, body = parse_request(conn)
        key = (method.upper(), path)
        log(f"İstek → {key}", YELLOW, "🔎")

        if key in ROUTES:
            handler = ROUTES[key]
            raw = global_error_handler(handler, headers, body) if method.upper() == "POST" else global_error_handler(handler)
        else:
            raw = global_error_handler(serve_static, path)

        status, ctype, resp_body = make_response_auto(raw)

    except Exception as e:
        log(f"Sunucu hatası: {e}", RED, "💥")
        status, ctype, resp_body = 500, "application/json", json.dumps({"error": str(e)}, ensure_ascii=False)
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        extra_headers["X-Response-Time"] = f"{elapsed_ms:.2f}ms"
        conn.sendall(build_http_response(status, ctype, resp_body, extra_headers))
        conn.close()
        color = GREEN if status < 400 else (YELLOW if status < 500 else RED)
        log(f"{status} → {path} ⏱ {elapsed_ms:.2f}ms", color, "✅" if status < 400 else ("⚠️" if status < 500 else "❌"))

# ==========================================================
# Sunucu Başlatma
# ==========================================================
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(64)
    log(f"Aşama 9.8 çalışıyor → http://{HOST}:{PORT}", CYAN, "🚀")

    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()
