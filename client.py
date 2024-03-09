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
        angle_valid = int(received_data.split(';')[0])
        distance_valid = int(received_data.split(';')[1])
        angle_deg = float(received_data.split(';')[2])  # angle is negative when the human is on the right of the pallet; positive for left side
        distance = float(received_data.split(';')[3])
        print("##############################")
        print("angle_valid:", angle_valid)
        print("distance_valid:", distance_valid)
        print("angle_deg:", angle_deg)
        print("distance:", distance)

    # Close the connection
    connection.close()

# Close the server socket (this line won't be reached in this example)
receiver_socket.close()
