import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from utils.ReadCameraModel import ReadCameraModel
from utils.UndistortImage import UndistortImage
import os
import csv

np.set_printoptions(suppress=True)

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('../Oxford_dataset/model/')
K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
H = np.identity(4)
N= 3871
final_points=[]
for i in range (20,N):
	print('frame',i)
	j=i+1
	img_current_frame = cv2.imread("../Visual-Odometry/FRAMES/{}.jpg".format(i))
	img_next_frame = cv2.imread("../Visual-Odometry/FRAMES/{}.jpg".format(j))
	img_current_frame = cv2.rectangle(img_current_frame,(np.float32(50),np.float32(np.shape(img_current_frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)
	img_next_frame = cv2.rectangle(img_next_frame,(np.float32(50),np.float32(np.shape(img_next_frame)[0])),(np.float32(1250),np.float32(800)),(0,0,0),-1)

	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img_current_frame,None)
	kp2, des2 = sift.detectAndCompute(img_next_frame,None)

	# FLANN parameters
	bf = cv2.BFMatcher()
	matches = bf.match(des1, des2)
	U = []
	V = []
	for m in matches:
		pts_1 = kp1[m.queryIdx]
		x1, y1 = pts_1.pt
		pts_2 = kp2[m.trainIdx]
		x2, y2 = pts_2.pt
		U.append((x1, y1))
		V.append((x2, y2))
	U = np.array(U)
	V = np.array(V)
	E, _ = cv2.findEssentialMat(U, V, focal=fx, pp=(cx, cy), method=cv2.RANSAC, prob=0.999, threshold=0.5)
	_, cur_R, cur_t, mask = cv2.recoverPose(E, U, V, focal=fx, pp=(cx, cy))
	if np.linalg.det(cur_R) < 0:
		cur_R = -cur_R
		cur_t = -cur_t
	new_pose = np.hstack((cur_R, cur_t))
	new_pose = np.vstack((new_pose, np.array([0, 0, 0, 1])))
	x1 = (H[0][3])
	z1 = (H[2][3])
	H = H@new_pose
	x = (H[0][3])
	z = (H[2][3])
	final_points.append([x1, x, z1, z])
	img_current_frame = cv2.resize(img_current_frame, (0, 0), fx=0.5, fy=0.5)
	plt.plot([x1, x], [-z1, -z], 'bo')
	if i % 50 == 0:
		plt.savefig("Output/plot-{}.png".format(i))
		np.save("Output/plotPoints_org.npy", final_points)
	plt.pause(0.01)
np.savetxt("points.csv", final_points, delimiter=",")
