import socket
import time

HOST = "127.0.0.1"
PORT = 8082
BUF = 4096

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((HOST, PORT))
    print("[CLIENT] Connected to server")

    msg = "selam gpt, veri kanal testi 🔁"
    data = msg.encode("utf-8")

    print(f"[CLIENT:SEND_RAW] {data}")
    client.sendall(data)
    print(f"[CLIENT:SENT] {msg}")

    # Sunucudan yanıtı al
    response = client.recv(BUF)
    print(f"[CLIENT:RECV_RAW] {response}")
    print(f"[CLIENT:RECV_DECODED] {response.decode('utf-8')}")

    client.close()
