# ==========================================================
# 🧩 AŞAMA 8 — Multi-Threaded Sunucu (Çoklu İstek)
# ==========================================================

import socket
import threading
import json

HOST = "127.0.0.1"
PORT = 8080
BUF = 4096


# ---------------------------
# Yardımcı Fonksiyonlar
# ---------------------------
def make_response(status, ctype, body):
    status_text = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
    response = f"""HTTP/1.1 {status} {status_text}
Content-Type: {ctype}; charset=utf-8
Content-Length: {len(body)}

{body}"""
    return response.encode("utf-8")


def parse_request(data):
    headers_part, _, body = data.partition("\r\n\r\n")
    lines = headers_part.split("\r\n")
    method, path, _ = lines[0].split()

    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()

    return method, path, headers, body


# ---------------------------
# Endpoint'ler
# ---------------------------
def handle_get_root():
    body = "<h1>Threaded API 🧠</h1><p>Her istek ayrı bir thread içinde çalışıyor!</p>"
    return 200, "text/html", body


def handle_post_sum(headers, body):
    if headers.get("content-type", "") != "application/json":
        err = json.dumps({"error": "Content-Type application/json olmalı"}, ensure_ascii=False)
        return 400, "application/json", err

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        err = json.dumps({"error": "Geçersiz JSON"}, ensure_ascii=False)
        return 400, "application/json", err

    if not all(k in data for k in ("a", "b")):
        err = json.dumps({"error": "JSON içinde 'a' ve 'b' anahtarları olmalı"}, ensure_ascii=False)
        return 400, "application/json", err

    try:
        a, b = float(data["a"]), float(data["b"])
        toplam = a + b
    except Exception:
        err = json.dumps({"error": "a ve b sayısal olmalı"}, ensure_ascii=False)
        return 400, "application/json", err

    result = json.dumps({"a": a, "b": b, "sum": toplam, "thread": threading.current_thread().name}, ensure_ascii=False)
    return 200, "application/json", result


# ---------------------------
# Router Tablosu
# ---------------------------
ROUTES = {
    ("GET", "/"): handle_get_root,
    ("POST", "/sum"): handle_post_sum,
}


# ---------------------------
# İstekleri işleyen thread fonksiyonu
# ---------------------------
def handle_client(conn, addr):
    print(f"📡 Yeni bağlantı: {addr} | Thread: {threading.current_thread().name}")

    data = b""
    while True:
        chunk = conn.recv(BUF)
        if not chunk:
            break
        data += chunk
        if b"\r\n\r\n" in data:
            break

    raw = data.decode("utf-8", errors="ignore")
    method, path, headers, body = parse_request(raw)

    key = (method, path)
    if key in ROUTES:
        if method == "GET":
            status, ctype, resp_body = ROUTES[key]()
        else:
            status, ctype, resp_body = ROUTES[key](headers, body)
    else:
        status, ctype, resp_body = 404, "text/html", "<h1>404 Not Found</h1>"

    response = make_response(status, ctype, resp_body)
    conn.sendall(response)
    conn.close()
    print(f"✅ {addr} isteği işlendi → {threading.current_thread().name}")


# ---------------------------
# Sunucu Başlatma (çok iş parçacıklı)
# ---------------------------
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(5)
    print(f"🚀 Multi-threaded API çalışıyor: http://{HOST}:{PORT}")

    while True:
        conn, addr = srv.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr))
        t.start()
