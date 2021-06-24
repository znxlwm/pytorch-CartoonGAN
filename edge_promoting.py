import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def edge_promoting(root, save, save_cropped):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    if not os.path.isdir(save_cropped):
        os.makedirs(save_cropped)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    n = 1
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(root, f))
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bigx, bigy, bigh, bigw = 0, 0, 0, 0

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if h > bigh or w > bigw:
                bigx = x
                bigy = y
                bigh = h
                bigw = w

        # ii = cv2.rectangle(rgb_img, (bigx, bigy), (bigx + bigw, bigy + bigh), (255, 0, 0), 3)

        cutoff = 10
        rgb_img = rgb_img[bigy + cutoff:bigy + bigh - cutoff, bigx + cutoff:bigx + bigw - cutoff]
        gray_img = gray_img[bigy + cutoff:bigy + bigh - cutoff, bigx + cutoff:bigx + bigw - cutoff]

        # DEBUGGING
        # _, ax = plt.subplots(3)
        # ax[0].imshow(thresh)
        # ax[1].imshow(ii)
        # ax[2].imshow(rgb_img)
        # plt.show()
        # print("bbox", bigx, bigy, bigh, bigw)

        rgb_img = cv2.resize(rgb_img, (256, 256))
        gray_img = cv2.resize(gray_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2, 2), (2, 2), (0, 0)), mode='reflect')
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        result = np.concatenate((rgb_img, gauss_img), 1)

        cv2.imwrite(os.path.join(save, str(n) + '.png'), result)
        cv2.imwrite(os.path.join(save_cropped, str(n) + '.png'), rgb_img)
        n += 1
