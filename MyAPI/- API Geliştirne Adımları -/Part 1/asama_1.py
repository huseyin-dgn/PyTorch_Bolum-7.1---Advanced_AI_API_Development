# ==========================================================
# 🧩 AŞAMA 1 — HTTP Metod Desteği (GET & POST)
# ==========================================================

import socket

HOST = "127.0.0.1"
PORT = 8080
BUF = 4096

# ---------------------------
# Route tablosu (method, path)
# ---------------------------
def handle_get_root():
    return 200, "text/html", "<h1>GET /</h1><p>Bu bir GET isteği.</p>"

def handle_post_root():
    return 200, "text/html", "<h1>POST /</h1><p>Bu bir POST isteği.</p>"

def handle_get_hello():
    return 200, "text/html", "<h1>Merhaba! 👋</h1><p>GET /hello isteği işlendi.</p>"

# Router sözlüğü
ROUTES = {
    ("GET", "/"): handle_get_root,
    ("POST", "/"): handle_post_root,
    ("GET", "/hello"): handle_get_hello,
}


# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def make_response(status_code, content_type, body):
    status_text = {
        200: "OK",
        404: "Not Found",
        405: "Method Not Allowed"
    }.get(status_code, "OK")

    response = f"""HTTP/1.1 {status_code} {status_text}
Content-Type: {content_type}; charset=utf-8
Content-Length: {len(body)}

{body}"""
    return response.encode("utf-8")

# ---------------------------
# Sunucu
# ---------------------------
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"🚀 Sunucu aktif: http://{HOST}:{PORT}")

    while True:
        client, addr = srv.accept()
        data = client.recv(BUF).decode("utf-8")

        if not data:
            client.close()
            continue

        # İlk satırı al: örn. "GET / HTTP/1.1"
        first_line = data.splitlines()[0]
        print(f"📩 İstek: {first_line}")

        try:
            method, path, _ = first_line.split()
        except ValueError:
            client.close()
            continue

        # Uygun route'u bul
        key = (method, path)
        if key in ROUTES:
            status, ctype, body = ROUTES[key]()
        else:
            # Aynı path varsa ama method yanlışsa 405
            allowed = [m for (m, p) in ROUTES if p == path]
            if allowed:
                status, ctype, body = 405, "text/html", f"<h1>405 Method Not Allowed</h1><p>İzin verilen: {', '.join(allowed)}</p>"
            else:
                status, ctype, body = 404, "text/html", "<h1>404 Not Found</h1>"

        # Yanıt gönder
        response = make_response(status, ctype, body)
        client.sendall(response)
        client.close()
