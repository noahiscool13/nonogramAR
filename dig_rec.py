import cv2
import numpy as np
digdict = [cv2.cvtColor(cv2.imread(f'digs/{n}.jpg'),cv2.COLOR_BGR2GRAY) for n in range(10)]
digdict = [cv2.threshold(x,127,255,cv2.THRESH_BINARY)[1] for x in digdict]

#
# test_set = [cv2.imread(f'test_digs/test{n}.jpg') for n in range(162)]
# test_set = [cv2.threshold(x,127,255,cv2.THRESH_BINARY)[1] for x in test_set]


def match(img):

    vals = [cv2.mean(cv2.absdiff(img,digdict[n]))[0] for n in range(10)]
    return min(range(10),key=lambda x:vals[x])


# for x in test_set:
#     print(match(x))
#     cv2.imshow("0",x)
#     cv2.waitKey(0)