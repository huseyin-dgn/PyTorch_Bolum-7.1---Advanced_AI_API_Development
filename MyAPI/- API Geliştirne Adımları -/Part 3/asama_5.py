# ==========================================================
# 🧩 AŞAMA 9.4.1 — Auto Response + Binary Static + Doğru Content-Length
# ==========================================================

import socket
import threading
import json
import os
import mimetypes
from typing import Any

# ---- Ayarlar ----
HOST = "127.0.0.1"
PORT = 8094
BUF = 4096

# Statik dosyaların bulunduğu klasör (senin verdiğin yol)
STATIC_DIR = r"C:\Users\hdgn5\OneDrive\Masaüstü\Kendi API'mız\- API Geliştirne Adımları -\Part 3"

mimetypes.init()

# ==========================================================
# 1) Dinamik Router (Decorator)
# ==========================================================
ROUTES = {}

def route(method: str, path: str):
    def decorator(func):
        ROUTES[(method.upper(), path)] = func
        print(f"📍 Route eklendi → {method.upper()} {path} -> {func.__name__}")
        return func
    return decorator

# ==========================================================
# 2) Auto Response (dict→JSON, str→HTML, bytes→binary)
# ==========================================================
def make_response_auto(data: Any):
    """Her türlü dönen veriyi (status,ctype,body) üçlüsüne normalize eder."""
    if isinstance(data, tuple):
        # Eski stil: (status, ctype, body)
        status, ctype, body = data
        return status, ctype, body

    if isinstance(data, dict):
        # Otomatik JSON
        return 200, "application/json", json.dumps(data, ensure_ascii=False)

    if isinstance(data, str):
        # Düz HTML/metin
        return 200, "text/html", data

    if isinstance(data, (bytes, bytearray)):
        # Binary içerik
        return 200, "application/octet-stream", bytes(data)

    # Bilinmeyen tür
    return 500, "application/json", json.dumps({"error": f"unsupported type: {type(data)}"}, ensure_ascii=False)

def build_http_response(status: int, ctype: str, body):
    """HTTP cevabını **bytes** olarak oluşturur (Content-Length byte cinsinden)."""
    status_text = {
        200: "OK",
        400: "Bad Request",
        404: "Not Found",
        500: "Internal Server Error"
    }.get(status, "OK")

    # body → bytes
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    elif isinstance(body, (bytes, bytearray)):
        body_bytes = bytes(body)
    else:
        # Emniyet: dict/list vs. gelirse JSON'a çevir
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        ctype = "application/json"

    headers = (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: {ctype}; charset=utf-8\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii")

    return headers + body_bytes

# ==========================================================
# 3) HTTP İstek Parsing (Content-Length destekli)
# ==========================================================
def _recv_until_headers_end(conn):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(BUF)
        if not chunk:
            break
        data += chunk
    return data

def parse_request(conn):
    """Header'ları okur, Content-Length varsa body'yi tam boyda çeker."""
    header_block = _recv_until_headers_end(conn)
    if not header_block:
        # Boş istek
        return "GET", "/", {}, ""

    headers_part, _, tail = header_block.partition(b"\r\n\r\n")
    head_lines = headers_part.decode("utf-8", errors="ignore").split("\r\n")

    # Request line
    method, path, _ = head_lines[0].split()
    # Headerlar
    headers = {}
    for line in head_lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    # Content-Length varsa geri kalan body'yi tamamla
    content_length = int(headers.get("content-length", "0")) if headers else 0
    body_bytes = tail
    to_read = max(0, content_length - len(body_bytes))
    while to_read > 0:
        chunk = conn.recv(min(BUF, to_read))
        if not chunk:
            break
        body_bytes += chunk
        to_read -= len(chunk)

    # Path normalize
    path = (path or "/").rstrip("/")  # "/sum/" → "/sum", "" → "/"
    if path == "":
        path = "/"

    try:
        body = body_bytes.decode("utf-8", errors="ignore")
    except:
        body = ""

    return method, path, headers, body

# ==========================================================
# 4) Endpoint’ler
# ==========================================================
@route("GET", "/")
def handle_root():
    return "<h1>Auto Response API ⚙️</h1><p>Handler'lar Python objesi döndürüyor; Content-Length byte cinsinden.</p>"

@route("POST", "/sum")
def handle_sum(headers, body):
    try:
        if headers.get("content-type", "").lower().startswith("application/json"):
            data = json.loads(body)
        else:
            return (400, "application/json", json.dumps({"error": "Content-Type application/json olmalı"}, ensure_ascii=False))
        a, b = float(data["a"]), float(data["b"])
        return {"a": a, "b": b, "sum": a + b, "thread": threading.current_thread().name}
    except Exception as e:
        return (400, "application/json", json.dumps({"error": str(e)}, ensure_ascii=False))

@route("GET", "/sum")
def handle_sum_get():
    # Basit bir HTML formu: tarayıcıdan test için kullanışlı
    return """<!DOCTYPE html>
<html lang="tr"><meta charset="utf-8"><title>Sum Test</title>
<body style="font-family:ui-sans-serif">
  <h3>Toplama (POST /sum)</h3>
  <input id="a" type="number" placeholder="a"> +
  <input id="b" type="number" placeholder="b">
  <button onclick="send()">Topla</button>
  <pre id="out"></pre>
  <script>
    async function send(){
      const a = parseFloat(document.getElementById('a').value||0);
      const b = parseFloat(document.getElementById('b').value||0);
      const res = await fetch('/sum', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({a,b})
      });
      document.getElementById('out').textContent = await res.text();
    }
  </script>
</body></html>"""


# ==========================================================
# 5) Statik Dosya Servisi (binary okuma + doğru MIME)
# ==========================================================
def serve_static(path: str):
    # / → varsayılan dosya
    if path in ("", "/"):
        path = "text.html"

    full_path = os.path.join(STATIC_DIR, path.lstrip("/"))

    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        return 404, "text/html", "<h1>404 Not Found</h1>"

    ctype, _ = mimetypes.guess_type(full_path)
    if not ctype:
        ctype = "application/octet-stream"

    with open(full_path, "rb") as f:
        body = f.read()

    return 200, ctype, body

def handle_client(conn, addr):
    print(f"📡 Yeni bağlantı: {addr} | Thread: {threading.current_thread().name}")
    try:
        method, path, headers, body = parse_request(conn)
        key = (method.upper(), path)
        print(f"🔎 İstek: {key}")

        if key in ROUTES:
            handler = ROUTES[key]
            raw = handler() if method == "GET" else handler(headers, body)
        else:
            raw = serve_static(path)

        status, ctype, resp_body = make_response_auto(raw)
    except Exception as e:
        print(f"⚠️ Hata: {e}")
        status, ctype, resp_body = 500, "application/json", json.dumps({"error": str(e)}, ensure_ascii=False)

    conn.sendall(build_http_response(status, ctype, resp_body))
    conn.close()
    print(f"✅ {addr} isteği işlendi → {threading.current_thread().name}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(64)
    print(f"\n🚀 Auto Response + Static Server çalışıyor: http://{HOST}:{PORT}\n")

    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()
