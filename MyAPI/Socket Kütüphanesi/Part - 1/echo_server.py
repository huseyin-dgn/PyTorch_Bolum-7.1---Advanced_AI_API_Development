import socket

HOST = "127.0.0.1"
PORT = 8082
BUF = 4096

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[SERVER] listening on {HOST}:{PORT}")

    while True:
        client_socket, client_addr = srv.accept()
        print(f"[+] Connected by {client_addr}")

        # Veri alma
        data = client_socket.recv(BUF)
        if not data:
            client_socket.close()
            continue

        print(f"[SERVER:RECV_RAW] {data}")
        print(f"[SERVER:RECV_DECODED] {data.decode('utf-8')}")

        # Veri gönderme
        client_socket.sendall(data)
        print(f"[SERVER:SENT] {data}")

        client_socket.close()
        print(f"[x] Connection closed: {client_addr}\n")