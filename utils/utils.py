import numpy as np

# Determining Homogenous Matrix
def HomogeneousMatrix(R, t):
    h = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    return h


def get_essential_matrix(K, F):
    E = K.T @ F @ K
    U, S, V_T = np.linalg.svd(E)

    E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
    return E


def ExtractCameraPose(E, K):
    """
    Args:
        E (array): Essential Matrix
        K (array): Intrinsic Matrix

    Returns:
        arrays: set of Rotation and Camera Centers
    """
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # print("E svd U", U)
    # print("E svd S", S)
    # print("E svd U[:, 2]", U[:, 2])
    R = []
    C = []
    P = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    return R, C
    #for i in range(4):
    #    if (np.linalg.det(R[i]) < 0):
    #        R[i] = -R[i]
    #        C[i] = -C[i]
    #    P.append(K @ R[i] @ np.hstack((np.eye(3), -C[i])))

    #return R, C, P;


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, x[0]], [x[1], x[0], 0]])


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):

    I = np.identity(3)
    sz = x1.shape[0]
    C1 = np.reshape(C1, (3, 1))
    C2 = np.reshape(C2, (3, 1))
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    #     print(P2.shape)
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


def DisambiguateCameraPose(Cset, Rset, Xset):
    """ Function to implement camera pose correction

    Args:
        Cset (TYPE): Set of calculated camera poses
        Rset (TYPE): Set of calculated Rotation matrices
        Xset (TYPE): 3D points

    Returns:
        TYPE: Corrected X, R_set, C_set
    """
    best = 0
    for i in range(4):

        #         Cset[i] = np.reshape(Cset[i],(-1,-1))
        N = Xset[i].shape[0]
        n = 0
        for j in range(N):
            if ((np.dot(Rset[i][2, :], (Xset[i][j, :] - Cset[i])) > 0)
                    and Xset[i][j, 2] >= 0):
                n = n + 1
        if n > best:
            C = Cset[i]
            R = Rset[i]
            X = Xset[i]
            best = n

    return X, R, C


def estimateFundamentalMatrix(points_a, points_b):

    points_num = points_a.shape[0]
    A = []
    B = np.ones((points_num, 1))

    cu_a = np.sum(points_a[:, 0]) / points_num
    cv_a = np.sum(points_a[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_a[:, 0] - cu_a)**2 + (points_a[:, 1] - cv_a)**2)**(1 / 2))
    T_a = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_a], [0, 1, -cv_a], [0, 0, 1]]))

    points_a = np.array(points_a.T)
    points_a = np.append(points_a, B)

    points_a = np.reshape(points_a, (3, points_num))
    points_a = np.dot(T_a, points_a)
    points_a = points_a.T

    cu_b = np.sum(points_b[:, 0]) / points_num
    cv_b = np.sum(points_b[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_b[:, 0] - cu_b)**2 + (points_b[:, 1] - cv_b)**2)**(1 / 2))
    T_b = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_b], [0, 1, -cv_b], [0, 0, 1]]))

    points_b = np.array(points_b.T)
    points_b = np.append(points_b, B)

    points_b = np.reshape(points_b, (3, points_num))
    points_b = np.dot(T_b, points_b)
    points_b = points_b.T

    for i in range(points_num):
        u_a = points_a[i, 0]
        v_a = points_a[i, 1]
        u_b = points_b[i, 0]
        v_b = points_b[i, 1]
        A.append([
            u_a * u_b, v_a * u_b, u_b, u_a * v_b, v_a * v_b, v_b, u_a, v_a, 1
        ])

    _, _, v = np.linalg.svd(A)
    F = v[-1]

    F = np.reshape(F, (3, 3)).T
    F = np.dot(T_a.T, F)
    F = np.dot(F, T_b)

    F = F.T
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = np.dot(U, S)
    F = np.dot(F, V)

    F = F / F[2, 2]

    return F

def GetInliersRANSAC(matches_a, matches_b):

    matches_num = matches_a.shape[0]
    Best_count = 0

    for iter in range(500):
        sampled_idx = np.random.randint(0, matches_num, size=8)
        F = estimateFundamentalMatrix(matches_a[sampled_idx, :],
                                      matches_b[sampled_idx, :])
        in_a = []
        in_b = []
        #ind = []
        update = 0
        for i in range(matches_num):
            matches_aa = np.append(matches_a[i, :], 1)
            matches_bb = np.append(matches_b[i, :], 1)
            error = np.dot(matches_aa, F.T)
            error = np.dot(error, matches_bb.T)
            if abs(error) < 0.005:
                in_a.append(matches_a[i, :])
                in_b.append(matches_b[i, :])
                #ind.append(indices[i])
                update += 1

        if update > Best_count:
            Best_count = update
            best_F = F
            inliers_a = in_a
            inliers_b = in_b
            #inlier_index = ind

    inliers_a = np.array(inliers_a)
    inliers_b = np.array(inliers_b)
    # inlier_index = np.array(inlier_index)

    return best_F, inliers_a, inliers_b
