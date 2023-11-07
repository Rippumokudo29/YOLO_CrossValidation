import os
import csv
import socket
import pandas as pd
from ultralytics import YOLO
import cv2
import glob

if __name__ == '__main__':

     INPUT_IMAGES = 'images/20231013_Yoshida/A2/exp0/stepA/stepA2/raw_imgs/'
     ROOT_OUTPUT_DIR = './runs/inference/'

     for exp in range(100):

          output_dir = ROOT_OUTPUT_DIR + f'exp{exp}/'
          # output_dir = ROOT_OUTPUT_DIR

          if not os.path.exists(output_dir):
               os.makedirs(output_dir, exist_ok=True)
               break

     MODEL_PATH =  "models/20231015_chojiku_MV_rotation_L/weights/best.pt"
     model = YOLO(MODEL_PATH)

     images = glob.glob(INPUT_IMAGES+'/*.jpg')

     for img in images:

          results = model.predict(source=img, 
                              project=output_dir,#保存先の上位ディレクトリ
                              name='images', #フォルダ名
                              exist_ok=True,
                              save=True,
                              conf=0.1)