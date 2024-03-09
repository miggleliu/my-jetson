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
from jetson_inference import poseNet
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
posenet = poseNet("resnet18-hand", threshold=0.15)
posenet_overlay = "links, keypoints"
# Get the total number of keypoints
num_keypoints = posenet.GetNumKeypoints()

# Get the names of each keypoint
for i in range(num_keypoints):
    keypoint_name = posenet.GetKeypointName(i)

    print(keypoint_name)
detectnet = detectNet("ssd-mobilenet-v2", threshold=0.5)
detectnet.SetOverlayAlpha(100)



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
    
    # ===== Use pose estimation to estimate distance ===== #
    # perform pose estimation (with overlay)
    poses = posenet.Process(undistorted_frame_cuda, overlay=posenet_overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    size = 0
    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)
        
        palm_idx = pose.FindKeypoint('palm')
        thumb_idx = pose.FindKeypoint('thumb_1')
        index_idx = pose.FindKeypoint('index_finger_1')
        middle_idx = pose.FindKeypoint('middle_finger_1')

        '''
        # find the keypoint index from the list of detected keypoints
        right_shoulder_idx = pose.FindKeypoint('right_shoulder')
        left_shoulder_idx = pose.FindKeypoint('left_shoulder')
        right_hip_idx = pose.FindKeypoint('right_hip')
        left_hip_idx = pose.FindKeypoint('left_hip')

        # if the keypoint index is < 0, it means it wasn't found in the image

        
        if left_hip_idx < 0 or left_shoulder_idx < 0:
            if right_hip_idx > 0 and right_shoulder_idx > 0:
                right_hip = pose.Keypoints[right_hip_idx]
                right_shoulder = pose.Keypoints[right_shoulder_idx]
                size = ((right_shoulder.x - right_hip.x)**2 + (right_shoulder.y - right_hip.y)**2)**0.5
            else:
                continue
        else:
            if right_hip_idx > 0 and right_shoulder_idx > 0:
                right_hip = pose.Keypoints[right_hip_idx]
                right_shoulder = pose.Keypoints[right_shoulder_idx]
                left_hip = pose.Keypoints[left_hip_idx]
                left_shoulder = pose.Keypoints[left_shoulder_idx]
                size = (((left_shoulder.x - left_hip.x)**2 + (left_shoulder.y - left_hip.y)**2)**0.5 + ((right_shoulder.x - right_hip.x)**2 + (right_shoulder.y - right_hip.y)**2)**0.5)/2
            else:
                left_hip = pose.Keypoints[left_hip_idx]
                left_shoulder = pose.Keypoints[left_shoulder_idx]
                size = ((left_shoulder.x - left_hip.x)**2 + (left_shoulder.y - left_hip.y)**2)**0.5
        print("size of body is", size)
        '''
    # ===== Use human detection to estimate angle ===== #
    detections = detectnet.Detect(undistorted_frame_cuda)
    detections_filtered = [d for d in detections if d.ClassID == 1]  # ClassID == 1 is for person class
    width_percentage = 0.5
    #height_percentage = 0.5
    for detection in detections_filtered:
        print("Class ID:", detection.ClassID)  # 1 for person
        print("Confidence:", detection.Confidence)
        print("Bounding Box:", detection.Left, detection.Top, detection.Right, detection.Bottom)  # top-left is (0,0), 640*480 max

        width_percentage = (detection.Left + detection.Right)/(2*width)
        #height_percentage = (detection.Top + detection.Bottom)/(2*height)
        print("width_percentage = ", width_percentage)

    # calculate distance and angle
    angle_rad = math.atan((1 - 2 * width_percentage) * math.tan(fov_width_rad))
    angle_deg = angle_rad * 180 / math.pi
    print("angle in degree = ", angle_deg)

    # Send the width percentage to receiver
    msg = str(width_percentage) + ';' + str(int(size))
    #sender_socket.send(msg.encode())

    display.Render(undistorted_frame_cuda)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(detectnet.GetNetworkFPS()))


# Close the socket
sender_socket.close()


