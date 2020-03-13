import cv2
import glob
import sys
import shutil
import os
import argparse
import numpy as np  


def main(threshold, source_dir):
    #サブディレクトリ一覧を取得
    sub_dirs = glob.glob(source_dir+"/**/")
    #各サブディレクトリについて
    progress_len = len(sub_dirs)
    progress_count = 0
    for sub_dir in sub_dirs:
        print(str(progress_count)+"/"+str(progress_len))
        progress_count += 1
        #隔離ディレクトリを作成
        os.makedirs(sub_dir+"/separate", exist_ok=True)
        #画像リストを生成
        img_list = make_direct_img_list(sub_dir)
        #距離行列を作成し距離が閾値より小さい画像を隔離
        separate_same_images(img_list, threshold)

def make_direct_img_list(sub_dir):
    img_list = glob.glob(sub_dir+"/*")
    img_list_buf = glob.glob(sub_dir+"/*")
    count = 0
    for i in range(len(img_list)):
        if not ".png" in img_list[i] and not ".jpg" in img_list[i]:
            img_list_buf.pop(i-count)
            count += 1
    return img_list_buf

def separate_same_images(img_list, threshold):
    sub_dir = img_list[0].rsplit("/",1)[0]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.AKAZE_create()
    l = len(img_list)
    for i in range(l):
        if not os.path.exists(img_list[i]):
            continue
        img_ref = cv2.imread(img_list[i])
        for j in range(i+1,l):
            if not os.path.exists(img_list[j]):
                continue
            img_tar = cv2.imread(img_list[j])
            distance = calc_distance(img_ref,img_tar,bf,detector)
            if distance < float(threshold):
                print("vvv same vvv")
                print(img_list[i])
                print(img_list[j])
                print(distance)
                print("^^^ ^^^^ ^^^")
                shutil.move(img_list[j],sub_dir+"/separate/"+os.path.basename(img_list[j]))

def calc_distance(img1,img2,bf,detector):
    (kp1, des1) = detector.detectAndCompute(img1, None)
    (kp2, des2) = detector.detectAndCompute(img2, None)
    
    matches = bf.match(des1, des2)
    dist = [m.distance for m in matches]
    ret = sum(dist) / len(dist)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='separating images program')
    parser.add_argument("threshold",help="the threshold to sepatrate", default="./scraped")
    parser.add_argument("-s","--source",help="the source directory", default="./scraped")
    args = parser.parse_args()    
    main(args.threshold,args.source)