# import re
from mss import mss
import cv2
from PIL import Image
import numpy as np
from time import time
import time
from scipy.spatial import distance as dist
# import grabscreen


sct = mss()


def grabthescreen():
    sct_img = sct.grab({'mon': 2, 'top': 150, 'left': 900,
                        'width': 400, 'height': 180})
    img = Image.frombytes(
        'RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return np.array(img_bgr)


def congray(colimage):
    gray = cv2.cvtColor(colimage, cv2.COLOR_BGR2GRAY)
    return gray

# ratheshv@hcl.com


def removeline(cap):
    thresh = cv2.threshold(
        cap, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # print(thresh)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(
        detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(cap, [c], -1, (255, 255, 255), 2)

    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = 255 - cv2.morphologyEx(255 - cap,
                                    cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    return result


def removenoise(grayimage):
    kernel = np.ones((5, 5), np.uint8)
    blur = cv2.GaussianBlur(grayimage, (9, 9), 1)
    thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)[1]
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opening


def filtercontours(grayimage):
    conlist = []
    returns, thresh = cv2.threshold(
        grayimage, 125, 255, cv2.THRESH_BINARY_INV)
    contours, hierachy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        area = cv2.contourArea(con)
        # print(area)
        if area > 150:
            conlist.append(con)

    return conlist


# starting
print("started")
for i in range(1, 5):
    print(i)
    time.sleep(1)

while True:
    cap = grabthescreen()
    # cap = cv2.imread("dinosmall.png")
    gray = congray(cap)
    result = removeline(gray)
    # # conv gray
    # grayCon = congray(result)
    # # remove noise
    removednoise = removenoise(result)
    # # contours drawing
    filterlist = filtercontours(removednoise)
    print(len(filterlist))

    # drawing contours

    if (filterlist == []):
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        for item in filterlist:
            M = cv2.moments(item)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # draw the contour and center of the shape on the image
            cv2.circle(cap, (cX, cY), 2, (0, 255, 0), -1)
        cv2.drawContours(cap, filterlist, -1, (0, 0, 255), 4)
        cv2.putText(cap, len(filterlist), ())
        cv2.imshow('image', cap)
        key = cv2.waitKey(1)
        if key == 27:
            break
