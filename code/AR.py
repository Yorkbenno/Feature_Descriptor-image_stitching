import numpy as np
import cv2
from PIL import Image
import planarH
import matplotlib.pyplot as plt


def compute_extrinsics(K, H):
    phi = np.linalg.inv(K) @ H
    phi_two_cloumn = phi[:, [0, 1]]
    U, L, VT = np.linalg.svd(phi_two_cloumn)
    middle_matrix = np.zeros((3, 2))
    middle_matrix[0][0] = 1
    middle_matrix[1][1] = 1
    rotation_two_column = U @ middle_matrix @ VT

    thrid_column = np.cross(rotation_two_column[:, 0], rotation_two_column[:, 1])
    thrid_column = thrid_column.reshape((1,3))
    R = np.hstack((rotation_two_column, np.transpose(thrid_column)))

    if np.linalg.det(R) == -1:
        R[:, 2] *= -1

    phi_last_column = phi[:, 2]

    val = 0
    for i in range(3):
        for j in range(2):
            val += phi[i][j] / R[i][j]

    val = val / 6
    t = phi_last_column.T / val

    return R, t


def project_extrinsics(K, W, R, t):
    t = np.reshape(t, (3,1))
    extrinsic = np.hstack((R, t))
    _, n = W.shape
    four_dimension_points = np.vstack((W, np.ones(n)))

    coor = K @ extrinsic @ four_dimension_points
    coor = coor / coor[2, :]

    return coor


if __name__ == '__main__':
    W = np.vstack(([0., 18.2, 18.2, 0.], [0., 0., 26., 26.], [0., 0., 0., 0.]))
    D = np.vstack(([483, 1704, 2175, 67], [810, 781, 2217, 2286]))
    K = np.vstack(([3043.72, 0.0, 1196.0], [0.0, 3043.72, 1604.0], [0.0, 0.0, 1.0]))
    W_2D = W[[0, 1], :]
    H3Dto2D = planarH.computeH(D, W_2D)

    R, t = compute_extrinsics(K, H3Dto2D)
    # Start process the data in txt
    f = open('../data/sphere.txt', 'r')
    content = f.readlines()
    f.close()
    # Now all the data are in memory so do not need to access the file
    # print(np.shape(content))
    xyz = []
    for dimension in content:
        data = dimension.split()
        points = [float(i) for i in data]
        xyz.append(points)

    xyz = np.vstack(xyz)
    # xyz[0, :] += 5
    # xyz[1, :] += 10
    # xyz[2, :] += 3

    coor = project_extrinsics(K, xyz, R, t)
    coor[0, :] += 340
    coor[1, :] += 570
    im = Image.open('../data/prince_book.jpeg')
    plt.imshow(im)

    plt.plot(coor[0, :], coor[1, :], 'y.', markersize=3)
    plt.draw()

    plt.savefig('../results/AR.png')
    plt.show()