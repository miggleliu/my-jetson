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
import math
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

# initial output for distance and angle
angle_rad = 0
angle_deg = 0
distance = 0

# Define the dimensions of the chessboard (number of inner corners)
num_corners_x = 9  # Number of inner corners along the horizontal axis
num_corners_y = 7  # Number of inner corners along the vertical axis

# --- set up TCP/IP connection with receiver ---
# Sender's IP address and port number
#receiver_ip = "192.168.1.108"
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

    # Convert the captured frame to grayscale
    grayscale_img = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    found, corners = cv2.findChessboardCorners(grayscale_img, (num_corners_x, num_corners_y))

    # If corners are found, draw them on the original frame
    if found:
        valid = 1
        cv2.drawChessboardCorners(undistorted_frame, (num_corners_x, num_corners_y), corners, found)
        distance1 = ((corners[0][0][0] - corners[8][0][0])**2 + (corners[0][0][1] - corners[8][0][1])**2) ** 0.5
        distance2 = ((corners[-9][0][0] - corners[-1][0][0])**2 + (corners[-9][0][1] - corners[-1][0][1])**2) ** 0.5
        distance = (distance1 + distance2) / 2
        distance = 100 / distance
        width_percentage = corners[31][0][0] / width
        angle_rad = math.atan((1 - 2 * width_percentage) * math.tan(fov_width_rad))
        angle_deg = angle_rad * 180 / math.pi
        print(distance)
        print(angle_deg)
    else:
        valid = 0
    # Convert the NumPy array to a CUDA image
    undistorted_frame_cuda = cudaFromNumpy(undistorted_frame)


    ## Send the width percentage to receiver
    msg = f"{valid};{valid};{angle_deg:.2f};{distance:.2f}"
    print("msg = ", msg)
    sender_socket.send(msg.encode())
    
    display.Render(undistorted_frame_cuda)
    #display.SetStatus("Object Detection | Network {:.0f} FPS".format(detectnet.GetNetworkFPS()))


# Close the socket
sender_socket.close()



