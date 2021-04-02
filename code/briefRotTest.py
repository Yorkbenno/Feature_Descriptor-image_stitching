import numpy as np
import cv2
import os
import BRIEF
import matplotlib.pyplot as plt
from scipy.spatial import distance

if __name__ == '__main__':
    im = cv2.imread('../data/model_chickenbroth.jpg')
    locs_base, desc_base = BRIEF.briefLite(im)
    width, height, channel = im.shape

    degrees = []
    correct = []
    for i in range(36):
        # Process for the degree and rotation
        degree = 10 * i
        degrees.append(degree)
        rotation_matrix = cv2.getRotationMatrix2D((height / 2, width / 2), degree, 1)
        rotation = cv2.warpAffine(im, rotation_matrix, (height, width))

        # Get the rotated image's locs and desc
        locs_rot, desc_rot = BRIEF.briefLite(rotation)
        match = BRIEF.briefMatch(desc_base, desc_rot)

        base_match = locs_base[match[:, 0], : -1]
        rot_match = locs_rot[match[:, 1], : -1]
        # base_match = base_match[:, [0, 1], : -1]
        # rot_match = rot_match[:, [0, 1]]

        base_transpose = np.transpose(base_match)
        print(base_transpose.shape)
        start = np.vstack((base_transpose, np.ones((1, base_transpose.shape[1]))))
        after = np.transpose(rotation_matrix.dot(start))

        count = 0
        for i in range(after.shape[0]):
            dis = distance.euclidean(after[i], rot_match[i])
            if dis < 10:
                count += 1

        # diff = np.linalg.norm(after - rot_match, axis=1)
        # count = diff[diff < 10]
        correct.append(count)

    plt.bar(degrees, correct, width=1.3)
    plt.xlabel('rotation_degree')
    plt.ylabel('num_correct')
    plt.title('Performance_analysis')
    plt.savefig('../results/rotation_performance')
    plt.show()
