import cv2
import glob
import random
import numpy as np
from utils import ReadCameraModel, UndistortImage


def estimateFundamentalMatrix(features):
    A = []
    #for x,y in
    return A


class Cells:
    def __init__(self):
        self.pts = list()
        self.pairs = dict()

    def rand_pt(self):
        return random.choice(self.pts)


if __name__=="__main__":
    img_path = "../Oxford_dataset/stereo/centre"
    images = glob.glob(img_path+"/*.png")
    print(len(images))
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    input_img = cv2.imread(images[0], 0)
    bgr_img = cv2.cvtColor(input_img, cv2.COLOR_BAYER_GR2BGR)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel('../Oxford_dataset/model/')
    undistorted_img = UndistortImage.UndistortImage(bgr_img, LUT)
    gray_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    y_bar, x_bar = np.array(gray_img.shape)/8

    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel.ReadCameraModel('../Oxford_dataset/model/')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    for i in range(1, len(images)):
        input_img_next = cv2.imread(images[i], 0)
        bgr_img_next = cv2.cvtColor(input_img_next, cv2.COLOR_BAYER_GR2BGR)
        undistorted_img_next = UndistortImage.UndistortImage(bgr_img_next, LUT)
        gray_img_next = cv2.cvtColor(undistorted_img_next, cv2.COLOR_BGR2GRAY)

        keypoints_next, descriptors_next = orb.detectAndCompute(gray_img_next, None)
        matches = bf.match(gray_img, gray_img_next)
        matches = sorted(matches, key=lambda x: x.distance)

        features, features_next = [], []
        for m in matches:
            features.append(keypoints[m.queryIdx].pt)
            features_next.append(keypoints_next[m.trainIdx].pt)


        matches_img = cv2.drawMatches(gray_img, keypoints, gray_img_next, keypoints_next, matches[:10], None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        #img = cv2.drawKeypoints(gray_img, keypoints_next, None)
        #cv2.imshow("keypoints", matches_img)
        #cv2.waitKey(1)

        # -----> Initialise the grids and points array variables <----- #
        point_correspondence_cf = np.zeros((len(matches),2))
        point_correspondence_nf = np.zeros((len(matches),2))
        grid = np.empty((8,8), dtype=object)
        grid[:,:] = Cells()

        for i, match in enumerate(matches):
            j = int(keypoints[match.queryIdx].pt[0]/x_bar)
            k = int(keypoints[match.queryIdx].pt[1]/y_bar)
            grid[j,k].pts.append(keypoints[match.queryIdx].pt)
            grid[j,k].pairs[keypoints[match.queryIdx].pt] = keypoints_next[match.trainIdx].pt

            point_correspondence_cf[i] = keypoints[match.queryIdx].pt[0], keypoints[match.queryIdx].pt[1]
            point_correspondence_nf[i] = keypoints_next[match.trainIdx].pt[0], keypoints_next[match.trainIdx].pt[1]

        F, mask = cv2.findFundamentalMat(point_correspondence_cf, point_correspondence_nf)				    # Estimate the Fundamental matrix #

        gray_img = gray_img_next
        keypoints = keypoints_next
        pass
    cv2.destroyAllWindows()
