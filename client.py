import socket
import json
import sys

HOST = "127.0.0.1"
PORT = 65432 

def client_init(clientID):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    with open(f"input{clientID}.txt", "r") as input_file:
        questions = input_file.read()
    
    client_socket.send(questions.encode('utf-8'))

    response = client_socket.recv(2048).decode('utf-8')
    # print(response[952])
    response_data = eval(response)
    # print(response_data)
    
    for response in response_data:
        response["ClientID"] = clientID
    
    with open(f"output{clientID}.json", "w") as output_file:
        output_file.write(json.dumps(response_data))

    client_socket.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # print("Usage: python client.py <client_id>")
        sys.exit(1)

    clientID = sys.argv[1]
    client_init(clientID)