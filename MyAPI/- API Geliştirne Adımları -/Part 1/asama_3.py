# ==========================================================
# 🧩 AŞAMA 7 — JSON Body (POST Verisi İşleme)
# ==========================================================

import socket
import json

HOST = "127.0.0.1"
PORT = 8080
BUF  = 4096

# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def make_response(status, content_type, body):
    """HTTP yanıtını formatlar."""
    status_text = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
    response = f"""HTTP/1.1 {status} {status_text}
Content-Type: {content_type}; charset=utf-8
Content-Length: {len(body)}

{body}"""
    return response.encode("utf-8")

def parse_request(data):
    """
    Gelen HTTP isteğini ayırır: başlıklar ve gövde.
    """
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
# Endpoint Fonksiyonları
# ---------------------------
def handle_get_root():
    """Basit GET endpoint."""
    body = "<h1>Mini API’ye hoş geldin 🧠</h1><p>POST /sum endpoint’ini dene!</p>"
    return 200, "text/html", body


def handle_post_sum(headers, body):
    """POST /sum → JSON verisini alır, toplar ve sonucu döner."""
    # Content-Type kontrolü
    if headers.get("content-type", "") != "application/json":
        err = json.dumps({"error": "Content-Type application/json olmalı"}, ensure_ascii=False)
        return 400, "application/json", err

    # Gövdeyi JSON’a çevir
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        err = json.dumps({"error": "Geçersiz JSON"}, ensure_ascii=False)
        return 400, "application/json", err

    # a ve b anahtarlarını kontrol et
    if not all(k in data for k in ("a", "b")):
        err = json.dumps({"error": "JSON içinde 'a' ve 'b' anahtarları olmalı"}, ensure_ascii=False)
        return 400, "application/json", err

    # İşlemi yap
    try:
        a, b = float(data["a"]), float(data["b"])
        toplam = a + b
    except Exception:
        err = json.dumps({"error": "a ve b sayısal olmalı"}, ensure_ascii=False)
        return 400, "application/json", err

    # Sonucu döndür
    result = json.dumps({"a": a, "b": b, "sum": toplam}, ensure_ascii=False)
    return 200, "application/json", result


# ---------------------------
# Router tablosu
# ---------------------------
ROUTES = {
    ("GET", "/"): handle_get_root,
    ("POST", "/sum"): handle_post_sum,
}


# ---------------------------
# Sunucu
# ---------------------------
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"🚀 JSON destekli mini API aktif: http://{HOST}:{PORT}")

    while True:
        conn, addr = srv.accept()
        data = b""
        while True:
            chunk = conn.recv(BUF)
            if not chunk:
                break
            data += chunk
            if b"\r\n\r\n" in data:
                # Başlıkları aldıktan sonra Content-Length kontrol et
                head, sep, rest = data.partition(b"\r\n\r\n")
                headers_text = head.decode("utf-8", errors="ignore")
                if "Content-Length" in headers_text:
                    for line in headers_text.split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            length = int(line.split(":")[1].strip())
                            break
                    # Gövde tamamlanana kadar okumaya devam et
                    while len(rest) < length:
                        rest += conn.recv(BUF)
                data = head + sep + rest
                break

        # İstek çözümleme
        raw = data.decode("utf-8", errors="ignore")
        method, path, headers, body = parse_request(raw)

        # Uygun route seçimi
        key = (method, path)
        if key == ("POST", "/sum"):
            status, ctype, response_body = handle_post_sum(headers, body)
        elif key == ("GET", "/"):
            status, ctype, response_body = handle_get_root()
        else:
            status, ctype, response_body = 404, "text/html", "<h1>404 Not Found</h1>"

        response = make_response(status, ctype, response_body)
        conn.sendall(response)
        conn.close()
