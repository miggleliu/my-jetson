#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import cv2
import numpy as np
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
from jetson_utils import cudaToNumpy, cudaFromNumpy
import socket


# --- Load the intrinsic parameters obtained from camera calibration ---
f_x = 725.55246441
c_x = 315.90664115
f_y = 677.78100733
c_y = 151.0298575

intrinsic_matrix = np.array([[f_x, 0, c_x],
                             [0, f_y, c_y],
                             [0, 0, 1]])

# --- Load the distortion coefficients obtained from camera calibration ---
k1 = -0.32027715
k2 = -0.05799121
p1 = 0.00167173
p2 = -0.00048511
k3 = 0.24113964
distortion_coeffs = np.array([k1, k2, p1, p2, k3])


# --- set up video ---
net = detectNet("ssd-mobilenet-v2", threshold=0.5)
net.SetOverlayAlpha(100)

#camera = videoSource("/dev/video0")  #csi://0   # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file

camera = cv2.VideoCapture(0)
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)    # 640
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 480

# --- calculate the Field of View (FOV) width using focal length x and resolution width ---
fov_width_rad = 2 * np.arctan(width / (2 * f_x))  # 0.830761110661
fov_width_deg = np.degrees(fov_width_rad)  # 47.5991054245

print("fov width rad = ", fov_width_rad)
print("fov width deg = ", fov_width_deg)

# --- set up TCP/IP connection with receiver ---
# Sender's IP address and port number
receiver_ip = "192.168.31.69"
receiver_port = 38584

# Create a TCP socket
sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the receiver
sender_socket.connect((receiver_ip, receiver_port))

# --- while loop for video streaming ---
while display.IsStreaming():
    
    ret, frame = camera.read()
    if not ret:
        break

    # apply intrinsic matrix and distortioin coefficients to the images
    undistorted_frame = cv2.undistort(frame, intrinsic_matrix, distortion_coeffs)

    # Convert the NumPy array to a CUDA image
    undistorted_frame_cuda = cudaFromNumpy(undistorted_frame)

    # Now 'undistorted_frame_cuda' contains the undistorted image on the GPU
    
    detections = net.Detect(undistorted_frame_cuda)
    detections_filtered = [d for d in detections if d.ClassID == 1]  # ClassID == 1 is for person class
    for detection in detections_filtered:
    #    print("Class ID:", detection.ClassID)  # 1 for person
    #    print("Confidence:", detection.Confidence)
    #    print("Bounding Box:", detection.Left, detection.Top, detection.Right, detection.Bottom)  # top-left is (0,0), 640*480 max

        width_percentage = "{:.4f}".format((detection.Left + detection.Right)/(2*width))
        height_percentage = "{:.4f}".format((detection.Top + detection.Bottom)/(2*height))
        print("width_percentage = ", width_percentage, "height_percentage = ", height_percentage)

        # Send the width percentage to receiver
        sender_socket.send(str(width_percentage).encode())

    display.Render(undistorted_frame_cuda)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))


# Close the socket
sender_socket.close()


