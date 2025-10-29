# ==========================================================
# 🧩 AŞAMA D — Mini Router Sistemi (Kendi API Çekirdeğimiz)
# ==========================================================

import socket
import json

HOST = "127.0.0.1"
PORT = 8080
BUF = 4096


def handle_request(path: str):
    """İstek yoluna göre uygun yanıtı döndürür."""
    if path == "/":
        body = "<h1>Ana Sayfa 🏠</h1><p>Hoş geldin, kendi API’m çalışıyor!</p>"
        content_type = "text/html"
    elif path == "/hello":
        body = "<h1>Merhaba! 👋</h1><p>Bu, kendi router’ımızdan dönen bir yanıt.</p>"
        content_type = "text/html"
    elif path == "/data":
        body = json.dumps(
            {"status": "ok", "message": "API’den selamlar", "version": 1.0},
            ensure_ascii=False
        )
        content_type = "application/json"
    else:
        body = "<h1>404 Not Found 🚫</h1><p>Bu sayfa bulunamadı.</p>"
        content_type = "text/html"

    return body, content_type


# ==========================================================
# 🌐 Sunucu başlatma
# ==========================================================
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"🚀 Sunucu dinlemede: http://{HOST}:{PORT}")

    while True:
        client_socket, client_addr = srv.accept()
        request = client_socket.recv(BUF).decode("utf-8")

        if not request:
            client_socket.close()
            continue

        # 1️⃣ HTTP isteğinin ilk satırını al ("GET /hello HTTP/1.1")
        request_line = request.splitlines()[0]
        print(f"📩 İstek Satırı: {request_line}")

        try:
            method, path, version = request_line.split()
        except ValueError:
            client_socket.close()
            continue

        # 2️⃣ İstek yoluna göre içerik oluştur
        body, content_type = handle_request(path)

        # 3️⃣ HTTP yanıtı oluştur
        response = f"""HTTP/1.1 200 OK
Content-Type: {content_type}; charset=utf-8
Content-Length: {len(body)}

{body}"""

        # 4️⃣ Yanıtı gönder ve bağlantıyı kapat
        client_socket.sendall(response.encode("utf-8"))
        client_socket.close()
