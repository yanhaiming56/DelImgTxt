import os

import cv2
import numpy as np

DATA_DIR = 'data'
IMG_DIR = 'img'
LABEL_DIR = 'label'
NEW_IMG_DIR = 'new_img'


def main(imgpath, labelpath):
    if os.path.exists(imgpath) == False:
        print("\033[1;31;40m%s doesn't exist!\033[0m" % (imgpath))
        return
    if os.path.exists(labelpath) == False:
        print("\033[1;31;40m%s doesn't exist!\033[0m" % (labelpath))
        return
    for imgfile in os.listdir(imgpath):
        imgfilepath = os.path.join(imgpath, imgfile)
        if os.path.isfile(imgfilepath) == False:
            continue
        pos = imgfile.rfind('.')
        imgname = imgfile[0:pos]
        imglabel = 'gt_' + imgname + '.txt'
        imglabelpath = os.path.join(labelpath, imglabel)
        if os.path.isfile(imglabelpath) == False:
            continue
        srcimg = cv2.imread(imgfilepath)

        with open(imglabelpath, 'r', encoding='utf-8', errors='ignore') as txt:
            for line in txt.readlines():
                srcmask = np.zeros(srcimg.shape[0:2], srcimg.dtype)
                linearr = line.strip().split(' ')
                linearr = np.array(linearr[0:4], np.int32)
                w = linearr[2] - linearr[0]
                h = linearr[3] - linearr[1]
                poly = np.array([[linearr[0], linearr[1]], [
                                linearr[0]+w, linearr[1]], [linearr[2], linearr[3]], [linearr[0], linearr[1]+h]])
                cv2.fillPoly(srcmask, [poly], (255, 255, 255))
                #cv2.rectangle(img,(linearr[0],linearr[1]),(linearr[2],linearr[3]),color = (0,0,255))
                srcimg = cv2.inpaint(srcimg, srcmask, 3, cv2.INPAINT_TELEA)
            newfilepath = os.path.join(DATA_DIR, NEW_IMG_DIR)
            if os.path.exists(newfilepath) == False:
                os.mkdir(newfilepath)
            newfilepath = os.path.join(newfilepath,imgfile)
            cv2.imwrite(newfilepath, srcimg)


if __name__ == '__main__':
    imgpath = os.path.join(DATA_DIR, IMG_DIR)
    labelpath = os.path.join(DATA_DIR, LABEL_DIR)
    main(imgpath, labelpath)