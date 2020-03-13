import cv2
import glob
import sys
import shutil
import os
import argparse
import numpy as np


def main(source_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    image_files = glob.glob(source_dir+"/*/*")
    index = 0
    os.makedirs(output_dir,exist_ok=True)

    for image_file in image_files:
        if (not (".png" in image_file)) and (not (".jpg" in image_file)):
            continue

        img = cv2.imread(image_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img_gray.shape

        faces = face_cascade.detectMultiScale(img_gray,minNeighbors=1,minSize=(100,100))
        
        for x,y,w,h in faces:
            if w < h:
                x = max(0,x - int((h-w)/2))
                w = min(h,width-x)
            else:
                y = max(0,y-int((w-h)/2))
                h = min(w,height-y)

            face_img = img[y:y+h,x:x+w]
            face_img = cv2.resize(face_img, (128,128))
            cv2.imwrite(output_dir+"/"+str(index)+".png",face_img)
            index += 1
        
        if index % 100 == 0:
            print(str(index) + "/" + str(len(image_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detecting faces program')
    parser.add_argument("-s","--source",help="the source directory", default="./scraped")
    parser.add_argument("-o","--output",help="the output directory", default="./face")
    parser.add_argument("-r","--reference",help="the feature reference", default="./lbpcascade_animeface.xml")
    args = parser.parse_args()    
    main(args.source,args.output)