# my-jetson
computer vision using Jetson Nano

1. run getimages.py to capture distorted images and save to image folder
2. run calibration.py to callibrate
3. run client.py to listen as a client (TCP/IP)
4. run server.py to parse video stream and send distances (in meter) and angles (in degree) to client

Only need to run 1 and 2 one time.

Note: configure your IP address using ifconfig in Linux terminal.
