import numpy as np
import cv2
import os
import math

if __name__ == '__main__':

    path = 'C:/Users/JEIMM/OneDrive/Escritorio/MMU-Iris-Database/1/left'
    file_name = 'aeval1.bmp'
    #path = '/Users/mbrav/OneDrive/Desktop/database/MMU-Iris-Database/3/right'
    #file_name = 'chingycr4.bmp'
    total_path = os.path.join(path, file_name)
    orig = cv2.imread(total_path)
    orig1 = cv2.imread(total_path,0)
    height, width = orig1.shape
    mask = np.zeros((height, width), np.uint8)

    # PUPILA
    img = orig.copy()
    img1 = orig.copy()
    img2 = orig.copy()
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.equalizeHist(im_gray)
    treshold = np.floor(np.max(im_gray) * 0.03)

    ret, im_bw = cv2.threshold(im_gray, treshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)

    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_OPEN, kernel)
    im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for idx, cont in enumerate(contours):
        cv2.drawContours(img1, contours, idx, (0, 0, 0), -1)
        cv2.drawContours(img2, contours, idx, (255, 255, 0), 2)
        M = cv2.moments(contours[idx])
        area = M['m00']
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        radius = int(np.ceil(np.sqrt(area / (2 * np.pi))))
        #cv2.circle(img, (cx, cy), 0, (255, 0, 0), 4)
        pupila = (cx, cy)
    h, w = orig.shape[:2]
    vis = np.zeros((h, w * 2 + 5), np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    vis[:h, :w] = orig
    vis[:h, w + 5:w * 2 + 5] = img2

    # IRIS
    detector = cv2.MSER_create()
    fs = detector.detect(im_gray)
    fs.sort(key=lambda x: -x.size)

    def supress(x):
        for f in fs:
            distx = f.pt[0] - x.pt[0]
            disty = f.pt[1] - x.pt[1]
            dist = math.sqrt(distx * distx + disty * disty)
            if (f.size > x.size) and (dist < f.size / 2):
                return True

    sfs = [x for x in fs if not supress(x)]

    for f in sfs:
        circles = cv2.circle(img2, (cx, cy), int(f.size / 2), (150, 55, 65), 2)
        circles = cv2.circle(img, (cx, cy), int(f.size / 2), (0, 0, 0), -1)
        cv2.circle(img2, (cx, cy), 0, (0, 0, 255), 4)
        break

    vis[:h, :w] = orig
    vis[:h, w + 5:w * 2 + 5] = img2
    print('center: ', (cx, cy))
    cv2.imshow('image', vis)
    cv2.waitKey(0)

    # IRIS EXTRACTION
    ret, im_bw = cv2.threshold(img, treshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.bitwise_and(im_bw,img1)
    cv2.imshow('eye', mask)
    cv2.waitKey(0)

    # POLAR COORDINATES
    img = mask.astype(np.float32)
    radius = int(f.size / 2)  # np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(img, (cx, cy), radius, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    cv2.imshow("Polar Image", polar_image)
    cv2.waitKey(0)