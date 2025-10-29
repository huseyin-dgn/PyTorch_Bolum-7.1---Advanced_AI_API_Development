# ==========================================================
# 🧩 AŞAMA 6 — Query String (URL Parametreleri)
# ==========================================================

import socket
from urllib.parse import urlsplit, parse_qs

HOST = "127.0.0.1"
PORT = 8080
BUF = 4096

# ---------------------------
# Yardımcı fonksiyonlar
# ---------------------------
def parse_request_line(request_line):
    """
    HTTP isteğinin ilk satırını (ör: 'GET /hello?name=Ebrar HTTP/1.1')
    parçalayarak method, path ve query verilerini döndürür.
    """
    method, target, _ = request_line.split()

    # urlsplit -> path ve query parçalarını ayırır
    parts = urlsplit(target)
    path = parts.path
    query = parse_qs(parts.query)  # {'name': ['Ebrar'], 'age': ['21']}

    # Tek değerli olanları sadeleştirelim:
    query = {k: v[0] if len(v) == 1 else v for k, v in query.items()}

    return method, path, query


def make_response(status, ctype, body):
    status_text = {200: "OK", 404: "Not Found"}.get(status, "OK")
    response = f"""HTTP/1.1 {status} {status_text}
Content-Type: {ctype}; charset=utf-8
Content-Length: {len(body)}

{body}"""
    return response.encode("utf-8")


# ---------------------------
# Route fonksiyonları
# ---------------------------
def handle_root(query):
    return 200, "text/html", "<h1>Ana Sayfa 🏠</h1><p>Hoş geldin!</p>"

def handle_hello(query):
    name = query.get("name", "Misafir")
    age = query.get("age")
    message = f"<h1>Merhaba, {name}! 👋</h1>"
    if age:
        message += f"<p>{age} yaşındasın!</p>"
    return 200, "text/html", message


# ---------------------------
# Router tablosu
# ---------------------------
ROUTES = {
    ( "GET", "/" ): handle_root,
    ( "GET", "/hello" ): handle_hello
}


# ---------------------------
# Sunucu
# ---------------------------
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"🚀 Query destekli API aktif: http://{HOST}:{PORT}")

    while True:
        client, addr = srv.accept()
        data = client.recv(BUF).decode("utf-8")

        if not data:
            client.close()
            continue

        # İlk satırı al
        first_line = data.splitlines()[0]
        print(f"📩 {first_line}")

        try:
            method, path, query = parse_request_line(first_line)
        except ValueError:
            client.close()
            continue

        key = (method, path)
        if key in ROUTES:
            status, ctype, body = ROUTES[key](query)
        else:
            status, ctype, body = 404, "text/html", "<h1>404 Not Found</h1>"

        response = make_response(status, ctype, body)
        client.sendall(response)
        client.close()
