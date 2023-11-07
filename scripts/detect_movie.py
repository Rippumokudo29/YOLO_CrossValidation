import os
import csv
import socket
import pandas as pd
from ultralytics import YOLO
import cv2
import glob
import time
import keyboard
import pathlib

def imgs_to_movie(dir_imgs, path_movie):

     img_array = []

     for filename in sorted(glob.glob(dir_imgs + "/*.jpg")):
          img = cv2.imread(filename)
          height, width, layers = img.shape
          size = (width, height)
          img_array.append(img)

     out = cv2.VideoWriter(path_movie, cv2.VideoWriter_fourcc(*'MP4V'), 15.0, size)

     for i in range(len(img_array)):
          out.write(img_array[i])
     out.release()

if __name__ == '__main__':

     INPUT_IMAGES = './input_images/1.apex_trial/'
     ROOT_OUTPUT_DIR = './runs/inference/'

     for exp in range(100):

          output_dir = ROOT_OUTPUT_DIR + f'exp{exp}/'
          # output_dir = ROOT_OUTPUT_DIR

          if not os.path.exists(output_dir):
               os.makedirs(output_dir, exist_ok=True)
               break
     
     dir_raw_img = output_dir + 'raw_img/'
     os.makedirs(dir_raw_img, exist_ok=True)

     MODEL_PATH =  './models/20230802_4chamber_v3_x/weights/best.pt'
     model = YOLO(MODEL_PATH)

     list_data = []
     
     cap = cv2.VideoCapture(0)
     img_num = 0
     label_det_list = ['img_num', 'conf_MV', 'conf_IVS', 'conf_TV']
     det_list = []

     try:
          while True:
               
               ret, frame = cap.read()

               cv2.imwrite(dir_raw_img+f'echo_{img_num:07}.jpg', frame)
               
               results = model.predict(source=frame,
                                        show=True)

               imageWidth = results[0].orig_shape[0]
               imageHeight = results[0].orig_shape[1]

               names = results[0].names
               classes = results[0].boxes.cls
               boxes = results[0].boxes
               annotatedFrame = results[0].plot()
               
               conf_MV = 0
               conf_IVS = 0
               conf_TV = 0
               
               for box, cls in zip(boxes, classes):
                    name = names[int(cls)]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    print(f"Object: {name} Conf={conf} Coordinates: StartX={x1}, StartY={y1}, EndX={x2}, EndY={y2}")

                    if name == 'MV' and conf > conf_MV:
                         conf_MV = conf
                    elif name == 'IVS' and conf > conf_IVS:
                         conf_IVS = conf
                    elif name == 'TV' and conf > conf_TV:
                         conf_TV = conf
               
               det_list.append([img_num, conf_MV, conf_IVS, conf_TV])
               
               img_num += 1
               
     except KeyboardInterrupt:
          pass

     df = pd.DataFrame(data=det_list, columns=label_det_list)
     path_csv = output_dir+'conf.csv'
     print(f'Resulut in saved to {path_csv}')
     df.to_csv(output_dir+'conf.csv', index=False)

     for img in glob.glob(dir_raw_img+'/*.jpg'):

          results = model.predict(source=img, 
                              project=output_dir,#保存先の上位ディレクトリ
                              name='images', #フォルダ名
                              exist_ok=True,
                              save=True,
                              show=False)