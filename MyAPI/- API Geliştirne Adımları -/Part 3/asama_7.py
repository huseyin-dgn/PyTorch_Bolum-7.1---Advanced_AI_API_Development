import socket , threading 
import json , os
import mimetypes
from typing import Any

HOST , PORT , BUF = "127.0.0.1" , 8092 , 4096

STATIC_DIR  = r"C:\Users\hdgn5\OneDrive\Masaüstü\Kendi API'mız\- API Geliştirne Adımları -\Part 3"

mimetypes.init()

ROUTES = {}

def route(method , path):
    def decorator(func):
        ROUTES[(method.upper() , path)] = func
        print(f"Route eklendi -> {method.upper()} {path} -> {func.__name__}")
        return func
    return decorator


def build_http_response(status: int, ctype: str, body):
    if isinstance(body, str):
        body_bytes = body.encode("utf-8")
    elif isinstance(body, (bytes, bytearray)):
        body_bytes = bytes(body)
    else:
        body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
        ctype = "application/json"

    status_text = {
        200: "OK", 400: "Bad Request", 404: "Not Found", 500: "Internal Server Error"
    }.get(status, "OK")

    headers = (
        f"HTTP/1.1 {status} {status_text}\r\n"
        f"Content-Type: {ctype}; charset=utf-8\r\n"
        f"Content-Length: {len(body_bytes)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode("ascii")

    return headers + body_bytes

def global_error_handler(func, *args):
    try:
        return func(*args)
    except FileNotFoundError as e:
        return (404, "application/json", json.dumps({"error": str(e), "type": "FileNotFoundError"}, ensure_ascii=False))
    except ValueError as e:
        return (400, "application/json", json.dumps({"error": str(e), "type": "ValueError"}, ensure_ascii=False))
    except KeyError as e:
        return (400, "application/json", json.dumps({"error": f"Eksik anahtar: {e}", "type": "KeyError"}, ensure_ascii=False))
    except Exception as e:
        return (500, "application/json", json.dumps({"error": str(e), "type": type(e).__name__}, ensure_ascii=False))
    
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
# 5️⃣ Endpoint’ler
# ==========================================================
@route("GET", "/")
def root():
    return "<h1>Global Error Handler ⚙️</h1><p>Artık her hata düzgün JSON formatında dönüyor!</p>"

@route("POST", "/sum")
def sum_numbers(headers, body):
    data = json.loads(body)
    a, b = float(data["a"]), float(data["b"])  # eksikse KeyError yakalanır
    return {"sum": a + b}

@route("POST", "/sum/plain")
def sum_plain(headers, body):
    data = json.loads(body)
    a, b = float(data["a"]), float(data["b"])
    return (200, "text/plain", str(a + b))

# ==========================================================
# 6️⃣ Statik Dosya Servisi
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

def handle_client(conn, addr):
    print(f"📡 Yeni bağlantı: {addr} | Thread: {threading.current_thread().name}")
    try:
        method, path, headers, body = parse_request(conn)
        key = (method.upper(), path)
        print(f"🔎 İstek: {key}")

        if key in ROUTES:
            handler = ROUTES[key]
            raw = global_error_handler(handler, headers, body) if method == "POST" else global_error_handler(handler)
        else:
            raw = global_error_handler(serve_static, path)

        status, ctype, resp_body = make_response_auto(raw)
    except Exception as e:
        status, ctype, resp_body = 500, "application/json", json.dumps({"error": str(e)}, ensure_ascii=False)

    conn.sendall(build_http_response(status, ctype, resp_body))
    conn.close()
    print(f"✅ {addr} isteği işlendi → {threading.current_thread().name}")

# ==========================================================
# 8️⃣ Sunucu Başlatma
# ==========================================================
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(64)
    print(f"\n🚀 Global Error Handler API çalışıyor: http://{HOST}:{PORT}\n")

    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()
