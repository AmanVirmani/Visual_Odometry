import cv2
import os
import argparse
import numpy as np
import glob
import random
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt

from Oxford_dataset.ReadCameraModel import ReadCameraModel
from Oxford_dataset.UndistortImage import UndistortImage

path = "Oxford_dataset/stereo/centre/*.png"
imagesPath = sorted(glob.glob(path))


fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./Oxford_dataset/model/')

# Camera Calibration Matrix of the model
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)


def estimateFundamentalMatrix(points_a,points_b):
    points_num = len(points_a)
    A = []

    for i in range(points_num):
        u_a = points_a[i][0]
        v_a = points_a[i][1]
        u_b = points_b[i][0]
        v_b = points_b[i][1]
        A.append([
            u_a * u_b, v_a * u_b, u_b, u_a * v_b, v_a * v_b, v_b, u_a, v_a, 1
        ])

    _, _, v = np.linalg.svd(A)
    F = v[-1].reshape(3,3)

    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = U @ S @ V

    return F

def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])

def rotationMatrixToEulerAngles(R) :
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

def drawCorrespondence(img1, img2, inliers_a, inliers_b):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2), dtype='uint8')
    out[:rows1, :cols1] = img1
    out[:rows2, cols1:cols1 + cols2] = img2
    radius = 4
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    thickness = 1

    assert len(inliers_a) == len(inliers_b), "inliers in images not equal"
    for m in range(0, len(inliers_a)):
        # Draw small circle on image 1
        cv2.circle(out, (int(inliers_a[m][0]), int(inliers_a[m][1])), radius,
                   RED, -1)

        # Draw small circle on image 2
        cv2.circle(out, (int(inliers_b[m][0]) + cols1, int(inliers_b[m][1])),
                   radius, GREEN, -1)

        # Draw line connecting circles
        cv2.line(out, (int(inliers_a[m][0]), int(inliers_a[m][1])),
                 (int(inliers_b[m][0]) + cols1, int(inliers_b[m][1])), GREEN,
                 thickness)

    return out

def getInliersRANSAC(features1,features2):
    noOfInliers = 0
    finalFundMatrix = np.zeros((3,3))

    inlier1 = [] 
    inlier2 = [] 

    # RANSAC Algorithm
    for i in range(0, 50): # 50 iterations for RANSAC 
        count = 0
        eightpoint = [] 
        goodFeatures1 = [] 
        goodFeatures2 = [] 
        tempfeature1 = [] 
        tempfeature2 = []
        
        while(True): 
            num = random.randint(0, len(features1)-1)
            if num not in eightpoint:
                eightpoint.append(num)
            if len(eightpoint) == 8:
                break

        for point in eightpoint: # Looping over eight random points
            goodFeatures1.append([features1[point][0], features1[point][1]]) 
            goodFeatures2.append([features2[point][0], features2[point][1]])
    
        # Computing Fundamentals Matrix from current frame to next frame
        FundMatrix = estimateFundamentalMatrix(goodFeatures1, goodFeatures2)

        for number in range(0, len(features1)):
            
            # If x2.T * F * x1 is less than threshold (0.01) then it is considered as Inlier
            matches_aa = np.array([features1[number][0],features1[number][1], 1]).T
            matches_bb = np.array([features2[number][0],features2[number][1], 1])
            error = np.squeeze(np.matmul((np.matmul(matches_bb,FundMatrix)),matches_aa))
            if abs(error) < 0.01:
                count = count + 1 
                tempfeature1.append(features1[number])
                tempfeature2.append(features2[number])

        if count > noOfInliers: 
            noOfInliers = count
            finalFundMatrix = FundMatrix
            inlier1 = tempfeature1
            inlier2 = tempfeature2

    return finalFundMatrix, inlier1, inlier2 

def calculateEssentialMatrix(K, F):
    E = np.dot(K.T, np.dot(F, K))
    U, S, V_T = np.linalg.svd(E)
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    return E

def estimateCameraPose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]
    
    return R,C

def linearTriangulation(K, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    sz = x1.shape[0]
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    X1 = np.hstack((x1, np.ones((sz, 1))))
    X2 = np.hstack((x2, np.ones((sz, 1))))

    X = np.zeros((sz, 3))

    for i in range(sz):
        skew1 = skew(X1[i, :])
        skew2 = skew(X2[i, :])
        A = np.vstack((np.dot(skew1, P1), np.dot(skew2, P2)))
        _, _, v = np.linalg.svd(A)
        x = v[-1] / v[-1, -1]
        x = np.reshape(x, (len(x), -1))
        X[i, :] = x[0:3].T

    return X

def disambiguateCameraPose(Cset, Rset, Xset):
    Cset = np.array(Cset)
    Rset = np.array(Rset)
    Xset = np.array(Xset)
    best = 0
    for i in range(4):
        angles = rotationMatrixToEulerAngles(Rset[i])
        N = Xset[i].shape[0]
        n = 0
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:
            for j in range(N):
                thirdrow = Rset[i][2,:].reshape((1,3))
                temp = Xset[i][j,:].reshape((3,1)) 
                if np.squeeze(thirdrow @ (temp - Cset[i])) > 0:
                    n = n + 1
            if n > best:
                C = Cset[i]
                R = Rset[i]
                X = Xset[i]
                best = n
        # else:
        #     print("here")
                if C[2] > 0:
                    C = -C

    return X, R, C

def NonLinearTriangulation(K, x1, x2, X_init, R1, C1, R2, C2):
    sz = x1.shape[0]
    assert x1.shape[0] == x2.shape[0] == X_init.shape[
        0], "2D-3D corresspondences have different shape "
    X = np.zeros((sz, 3))

    init = X_init.flatten()
    #     Tracer()()
    optimized_params = opt.least_squares(
        fun=minimizeFunction,
        x0=init,
        method="dogbox",
        args=[K, x1, x2, R1, C1, R2, C2])

    X = np.reshape(optimized_params.x, (sz, 3))

    return X


def minimizeFunction(init, K, x1, x2, R1, C1, R2, C2):
    sz = x1.shape[0]
    X = np.reshape(init, (sz, 3))

    I = np.identity(3)
    C2 = np.reshape(C2, (3, -1))

    X = np.hstack((X, np.ones((sz, 1))))

    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    error1 = 0
    error2 = 0
    error = []

    u1 = np.divide((np.dot(P1[0, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    v1 = np.divide((np.dot(P1[1, :], X.T).T), (np.dot(P1[2, :], X.T).T))
    u2 = np.divide((np.dot(P2[0, :], X.T).T), (np.dot(P2[2, :], X.T).T))
    v2 = np.divide((np.dot(P2[1, :], X.T).T), (np.dot(P2[2, :], X.T).T))

    #     print(u1.shape,x1.shape)
    assert u1.shape[0] == x1.shape[0], "shape not matched"

    error1 = ((x1[:, 0] - u1) + (x1[:, 1] - v1))
    error2 = ((x2[:, 0] - u2) + (x2[:, 1] - v2))
    #     print(error1.shape)
    error = sum(error1, error2)

    return sum(error)

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="./Oxford_dataset/", help='Folder of dataset')
    Parser.add_argument('--loadData',default=True, help='Load presaved data to run the code quickly')
    Parser.add_argument('--visualize',default=False,help='Set flag for visualization purposes')
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    Args = Parser.parse_args()
    DataPath = Args.DataPath
    inbuiltFunc = False
    loadData = Args.loadData
    visualize = Args.visualize

    # Initial camera Pose is considered as an identity matrix
    H = np.identity(4)  
    origin = np.array([[0, 0, 0, 1]]).T
    sift = cv2.xfeatures2d.SIFT_create() 
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    trajPts = [] 
    matchImages = []
    
    if(loadData):
        FList = np.load("FList.npy")
        inlier1List = np.load("inlier1List.npy")
        inlier2List = np.load("inlier2List.npy")

    # skippping first 20 frames
    for ind in range(20,len(imagesPath)-1):  

        print(ind)

        img1 = cv2.imread(imagesPath[ind], 0)  
        rgbImg1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
        undistortImage1 = UndistortImage(rgbImg1, LUT)  
        gray1 = cv2.cvtColor(undistortImage1, cv2.COLOR_BGR2GRAY)

        img2 = cv2.imread(imagesPath[ind+1], 0)  
        rgbImg2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
        undistortImage2 = UndistortImage(rgbImg2, LUT)  
        gray2 = cv2.cvtColor(undistortImage2, cv2.COLOR_BGR2GRAY)

        # Cropping the area of interest 
        # This is used because the car will be there in all the frames which will cause issue in finding the tranformation in two consecutive frames.. 
        # pixels associated will car will lead to no rotation and translation between two consecutive frames
        gray1Crop = gray1[200:650, 0:1280]
        gray2Crop = gray2[200:650, 0:1280]

        if(not loadData):
            kp1, des1 = sift.detectAndCompute(gray1Crop, None)
            kp2, des2 = sift.detectAndCompute(gray2Crop, None)
            matches = flann.knnMatch(des1, des2, k=2)

            good = []
            points_a = []
            points_b = []

            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    good.append(m)
                    points_b.append(kp2[m.trainIdx].pt)
                    points_a.append(kp1[m.queryIdx].pt)

            points_a = np.int32(points_a)
            points_b = np.int32(points_b)

            F, inliers_a, inliers_b = getInliersRANSAC(points_a,points_b)

            if(visualize):
                matchesImg = cv2.drawMatches(gray1Crop, kp1, gray2Crop, kp2, good, None, flags=2)
                matchImages.append(matchesImg)
                cv2.namedWindow("matchesImg",cv2.WINDOW_NORMAL)
                cv2.resizeWindow("matchesImg",(800,800))
                cv2.imshow("matchesImg",matchesImg)
                # outputImg = 'Output/matchesImg.png'
                # cv2.imwrite(outputImg,matchesImg)
                cv2.waitKey(1000)

        else:    
            F = FList[ind-20]
            inliers_a = inlier1List[ind-20]
            inliers_b = inlier2List[ind-20]


        if(visualize):
            ## Drawing Inliers
            inlierImg = drawCorrespondence(gray1Crop,gray2Crop,inliers_a,inliers_b)
            cv2.namedWindow("inlierImg",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("inlierImg",(800,800))
            cv2.imshow("inlierImg",inlierImg)
            # outputImg = 'Output/inliersImg.png'
            # cv2.imwrite(outputImg,inlierImg)
            cv2.waitKey(100)         

        
        # Finding Essential Matrix
        E = calculateEssentialMatrix(K, F)
        # Finding 4 set of solutions of rotation matrix and translation vector
        Rset, Tset = estimateCameraPose(E)
        Tset = np.reshape(Tset, (4, 3, 1))

        # List for 4set of 3d points considering each rotation and translation
        Xset = []
        for n in range(0, 4):
            X1 = linearTriangulation(K, np.zeros((3, 1)), np.identity(3),
                                     Tset[n].T, Rset[n], np.float32(inliers_a),
                                     np.float32(inliers_b))
            Xset.append(X1)
        X, R, T = disambiguateCameraPose(Tset, Rset, Xset)

        X = NonLinearTriangulation(K, np.float32(inliers_a), np.float32(inliers_b), X,
                               np.eye(3), np.zeros((3, 1)), R, T)
        print('Non Linear Triangulation Output - ', X)

        

        # Transforming from current frame to next frame
        Rt = np.column_stack((R, T)) 
        matHomogeneous = np.vstack((Rt, np.array([0, 0, 0, 1])))
        H = H @ matHomogeneous
        p = H @ origin 

        trajPts.append([p[0][0], -p[2][0]])
        print(trajPts[ind-20])

        plt.scatter(p[0][0], -p[2][0], color='r')
    np.save('trajCustom',trajPts)
    np.save('matchImages',matchImages)    
    plt.show()





if __name__ == '__main__':
    main()
