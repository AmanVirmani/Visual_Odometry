import cv2
import numpy as np


def get_camera_pose(E, U, V, K):
    R, C = ExtractCameraPose(E, K)

    count_numbers = []
    count_numbers.append(count_positive_depth_points(R[0], C[0], U, V, K))
    count_numbers.append(count_positive_depth_points(R[1], C[1], U, V, K))
    count_numbers.append(count_positive_depth_points(R[2], C[2], U, V, K))
    count_numbers.append(count_positive_depth_points(R[3], C[3], U, V, K))

    index = np.argmax(count_numbers)

    return R[index], C[index]


def getFundamentalMatrix(U, V, max_iters=500, threshold=0.008):
    Inliers_UN = []
    max_inliers = 0

    ## get fundamental matrix
    for it in range(max_iters):
        X1,X2 = get_random_matches(U,V)
        F_r = get_normalized_fundamental_matrix(X1,X2)
        Inliers_U = []
        Inliers_V = []
        Inliers = 0
        for j in range(len(U)):
            U1 = np.array([U[j][0], U[j][1], 1]).reshape(1, -1)
            V1 = np.array([V[j][0], V[j][1], 1]).reshape(1, -1)

            epiline1 = F_r @ U1.T
            epiline2 = F_r.T @ V1.T
            error_bottom = epiline1[0]**2 + epiline1[1]**2 + epiline2[0]**2 + epiline2[1]**2

            error = ((V1 @ F_r @ U1.T)**2)/error_bottom

            if error[0, 0] < threshold:
                Inliers += 1
                Inliers_U.append([U[j][0], U[j][1]])
                Inliers_V.append([V[j][0], V[j][1]])

        if max_inliers < Inliers:
            max_inliers = Inliers
            Inliers_UN = Inliers_U
            Inliers_VN = Inliers_V
            F = F_r
    return F


def get_random_matches(points,points_next):
    random_indices = np.random.choice(len(points), size=8, replace=False)
    return points[random_indices, :], points_next[random_indices, :]


def get_normalized_fundamental_matrix(points1, points2):

    dist1 = np.sqrt((points1[:,0]- np.mean(points1[:,0]))**2 + (points1[:,1]- np.mean(points1[:,1]))**2)
    dist2 = np.sqrt((points2[:,0]- np.mean(points2[:,0]))**2 + (points2[:,1]- np.mean(points2[:,1]))**2)

    m_dist1 = np.mean(dist1)
    m_dist2 = np.mean(dist2)

    scale1 = np.sqrt(2)/m_dist1
    scale2 = np.sqrt(2)/m_dist2

    t1 = np.array([[scale1, 0, -scale1*np.mean(points1[:,0])],[0, scale1, -scale1*np.mean(points1[:,1])],[0, 0, 1]])
    t2 = np.array([[scale2, 0, -scale2*np.mean(points2[:,0])],[0, scale2, -scale2*np.mean(points2[:,1])],[0, 0, 1]])


    U_x = (points1[:,0] - np.mean(points1[:,0]))*scale1
    U_y = (points1[:,1] - np.mean(points1[:,1]))*scale1
    V_x = (points2[:,0] - np.mean(points2[:,0]))*scale2
    V_y = (points2[:,1] - np.mean(points2[:,1]))*scale2

    A = np.zeros((len(U_x),9))

    for i in range(len(U_x)):

        A[i] = np.array([U_x[i]*V_x[i], U_y[i]*V_x[i], V_x[i], U_x[i]*V_y[i], U_y[i]*V_y[i], V_y[i], U_x[i], U_y[i], 1])

    U,S,V = np.linalg.svd(A)
    V = V.T
    F = V[:,-1].reshape(3,3)

    # Enforcing rank 2 constraint on estimated F matrix
    Uf, Sf, Vf = np.linalg.svd(F)
    SF = np.diag(Sf)
    SF[2,2] = 0

    F = Uf @ SF @ Vf
    F = t2.T @ F @ t1
    F = F/F[2, 2]

    return F


def getEssentialMatrix(F, K):
    E = K.T @ F @ K

    U, S, V_T = np.linalg.svd(E)
    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    E = E/np.linalg.norm(E)
    return E


def ExtractCameraPose(E, K):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    P = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2].reshape(-1, 1))
    C.append(-U[:, 2].reshape(-1, 1))
    C.append(U[:, 2].reshape(-1, 1))
    C.append(-U[:, 2].reshape(-1, 1))

    for i in range(len(R)):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]
    return R, C


def LinearTriangulation(P1, P2, pts1, pts2):
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))

    x = np.zeros((3, len(pts1)))

    pt1 = np.array(-np.eye(2, 3))
    pt2 = np.array(-np.eye(2, 3))

    for i in range(len(pts1)):
        pt1[:, 2] = pts1[i, :]
        pt2[:, 2] = pts2[i, :]

        A[0:2,:] = pt1.dot(P1[0:3,0:3])
        A[2:4,:] = pt2.dot(P2[0:3,0:3])

        b[0:2,:] = pt1.dot(P1[0:3,3:4])
        b[2:4,:] = pt2.dot(P2[0:3,3:4])

        cv2.solve(A,b,x[:,i:i+1],cv2.DECOMP_SVD)

    return x

def count_positive_depth_points(R, C, pts1, pts2, K):

    P1 = np.eye(3, 4)
    P2 = np.hstack((R , C.reshape(-1, 1)))

    X1_p = np.array(LinearTriangulation(P1, P2, np.array(pts1), np.array(pts2)))

    X1 = np.vstack((X1_p,np.ones((1,len(pts1))))).reshape(-1,4)

    X1 = np.divide(X1,np.array([X1[:,3],X1[:,3],X1[:,3],X1[:,3]]).T)

    X1 = np.sum(P2@X1.T > 0)

    return X1
