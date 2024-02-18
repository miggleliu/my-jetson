import socket

# Receiver's IP address and port number
receiver_ip = "192.168.31.69"
receiver_port = 38584

# Create a TCP server socket
receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the receiver's IP address and port number
receiver_socket.bind((receiver_ip, receiver_port))

# Listen for incoming connections
receiver_socket.listen()

print("Receiver is listening...")

while True:
    # Accept incoming connections
    connection, address = receiver_socket.accept()
    print("Connected to:", address)
    
    # Keep receiving data
    while True:
        # Receive data
        received_data = connection.recv(1024).decode()

        if not received_data:
            # If no data received, the connection might have been closed by the sender
            print("Connection closed by the sender.")
            break

        # Process the received data (convert it back to integer)
        received_number = float(received_data)
        print("Received number:", received_number)

    # Close the connection
    connection.close()

# Close the server socket (this line won't be reached in this example)
receiver_socket.close()
