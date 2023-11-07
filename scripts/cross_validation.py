import os
import pandas as pd
from ultralytics import YOLO
import cv2
import glob
import random
import pathlib
import shutil

def setting(root_dir_dataset, num_division, output_dir):
    
    dir_test_imgs = root_dir_dataset+'/test/images/'
    dir_label_imgs = root_dir_dataset+'/test/labels/'
    os.makedirs(dir_test_imgs, exist_ok=True)
    os.makedirs(dir_label_imgs, exist_ok=True)
    
    df_dataset = pd.DataFrame(data=None, columns=['image_id', 'train0_val1', 'batch_num'])
    
    dir_images = root_dir_dataset + '/images/'
    
    path_train_imgs = pathlist_to_namelist(glob.glob(dir_images+'/train/*.jpg'))
    random.shuffle(path_train_imgs)
    nums_per_train_batch = int(len(path_train_imgs)/num_division)
    nums_remainder_train_batch = len(path_train_imgs)%num_division
    
    train_batchs = [path_train_imgs[i*nums_per_train_batch:(i+1)*nums_per_train_batch] for i in range(num_division)]
    
    for remainder in range(nums_remainder_train_batch):
        
        train_batchs[-1].append(path_train_imgs[-(remainder+1)])
    
    for batch_num, batch_data in enumerate(train_batchs):
        
        for data in batch_data:
            
            df_dataset.loc[len(df_dataset)] = [data, 0, batch_num]
    
    path_val_imgs = pathlist_to_namelist(glob.glob(dir_images+'/val/*.jpg'))
    random.shuffle(path_val_imgs)
    nums_per_val_batch = int(len(path_val_imgs)/num_division)
    nums_remainder_val_batch = len(path_val_imgs)%num_division
    
    val_batchs = [path_val_imgs[i*nums_per_val_batch:(i+1)*nums_per_val_batch] for i in range(num_division)]
    
    for remainder in range(nums_remainder_val_batch):
        
        train_batchs[-1].append(path_val_imgs[-(remainder+1)])
    
    for batch_num, batch_data in enumerate(val_batchs):
        
        for data in batch_data:
            
            df_dataset.loc[len(df_dataset)] = [data, 1, batch_num]
    
    path_infocsv = output_dir+'InfoDataset.csv'
    df_dataset.to_csv(path_infocsv, index=False)
    
    return df_dataset


def pathlist_to_namelist(pathlist):
    
    namelist = []
    
    for path in pathlist:
        
        namelist.append(str(pathlib.Path(path).name))
    
    return namelist


def split_data(root_dir_dataset, batch_num, df_info):
    
    dir_test_imgs = root_dir_dataset+'test/images/'
    os.makedirs(dir_test_imgs, exist_ok=True)
    dir_test_labels = root_dir_dataset+'test/labels/'
    os.makedirs(dir_test_labels, exist_ok=True)
    dir_train_imgs = root_dir_dataset + 'images/train/'
    dir_train_labels = root_dir_dataset + 'labels/train/'
    dir_val_imgs = root_dir_dataset + 'images/val/'
    dir_val_labels = root_dir_dataset + 'labels/val/'
    
    df_test_imgs = df_info[df_info['batch_num']==batch_num]
    
    for index, data in df_test_imgs.iterrows():
        
        if data[1] == 0:
            shutil.move(dir_train_imgs+f'{data[0]}', dir_test_imgs)
            shutil.move(dir_train_labels+f'{os.path.splitext(data[0])[0]}.txt', dir_test_labels)
        elif data[1] == 1:
            shutil.move(dir_val_imgs+f'{data[0]}', dir_test_imgs)
            shutil.move(dir_val_labels+f'{os.path.splitext(data[0])[0]}.txt', dir_test_labels)


def gather_data_into_trainval(root_dir_dataset, df_info):
    
    dir_test_imgs = root_dir_dataset+'test/images/'
    dir_test_labels = root_dir_dataset+'test/labels/'
    
    dir_train_imgs = root_dir_dataset + 'images/train/'
    dir_train_labels = root_dir_dataset + 'labels/train/'
    dir_val_imgs = root_dir_dataset + 'images/val/'
    dir_val_labels = root_dir_dataset + 'labels/val/'
    
    for test_img in glob.glob(dir_test_imgs+'/*'):
        
        train0_val1 = int(df_info[df_info['image_id']==str(pathlib.Path(test_img).name)]['train0_val1'])
        
        if train0_val1 == 0:
            
            shutil.move(test_img, dir_train_imgs)
            shutil.move(dir_test_labels+f'{os.path.splitext(pathlib.Path(test_img).name)[0]}.txt', dir_train_labels)
        
        elif train0_val1 == 1:
            
            shutil.move(test_img, dir_val_imgs)
            shutil.move(dir_test_labels+f'{os.path.splitext(pathlib.Path(test_img).name)[0]}.txt', dir_val_labels)


def train(yaml_path, batch_num, output_dir):
    
    model = YOLO("yolov8n.pt")

    dir_model_save = output_dir+'results/'
    os.makedirs(dir_model_save, exist_ok=True)
    
    model.train(data=yaml_path, epochs=150, batch=2**3, workers=1, degrees=180, pretrained=True, project=dir_model_save, name=f'batch{batch_num}')
    
    model_path = dir_model_save + f'/batch{batch_num}/weights/best.pt'
    
    return model_path


def evaluate(copied_dir, model_path, iou_threshold, list_classes, conf_threshold, batch_output_dir):
    
    dir_test_imgs = copied_dir + 'test/images/'
    dir_test_labes = copied_dir + 'test/labels/'
    
    dir_yolo_imgs = batch_output_dir + 'predicted_imgs/'
    os.makedirs(dir_yolo_imgs, exist_ok=True)
    
    model = YOLO(model_path)
    
    df_confusion_matrix = pd.DataFrame(data=None, columns=list_classes+['FalseNegative'])
    for class_name in list_classes:
        df_confusion_matrix.loc[class_name] = [0 for i in range(len(list_classes)+1)]
    df_confusion_matrix.loc['FalsePositive'] = [0 for i in range(len(list_classes)+1)]
    
    for img in glob.glob(dir_test_imgs+'/*'):
        
        img_name = str(pathlib.Path(img).name)
        label_path = dir_test_labes+f'{os.path.splitext(img_name)[0]}.txt'
        
        results = model.predict(source=img, 
                            save=False,
                            conf=conf_threshold)
        
        imageHeight = results[0].orig_shape[0]
        imageWidth = results[0].orig_shape[1]
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes
        annotatedFrame = results[0].plot()
        cv2.imwrite(dir_yolo_imgs+img_name, annotatedFrame)
        
        list_predicted_data = []
        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            
            list_predicted_data.append([LIST_CLASSE.index(name), 
                                        x1/imageWidth, y1/imageHeight, x2/imageWidth, y2/imageHeight])
        
        list_true_data = []
        with open(label_path) as f:
            for line in f:
                data = line.split()
                true_cls = int(data[0])
                true_xc = float(data[1])
                true_yc = float(data[2])
                true_width = float(data[3])
                true_height = float(data[4])
                list_true_data.append([true_cls, true_xc-true_width/2, true_yc-true_height/2, true_xc+true_width/2, true_yc+true_height/2])
        f.close()
        
        for index_true, data_true in enumerate(list_true_data):
            
            max_iou = 0
            correspond_index = None
            
            for index_predicted, data_predicted in enumerate(list_predicted_data):
                
                if data_predicted[0] == data_true[0]:
                    iou = calc_iou(data_true, data_predicted)
                    if (iou > max_iou) and (iou > iou_threshold):
                        max_iou = iou
                        correspond_index = index_predicted
            
            if correspond_index is not None:
                
                print(f'Correct {list_classes[data_true[0]]}')
                df_confusion_matrix.at[f'{list_classes[data_true[0]]}', f'{list_classes[data_true[0]]}'] += 1
                
                del list_true_data[index_true]
                del list_predicted_data[correspond_index]
        
        for data_true in list_true_data:
            
            df_confusion_matrix.at['FalsePositive', f'{list_classes[data_true[0]]}'] += 1
        
        for data_predicted in list_predicted_data:
            
            df_confusion_matrix.at[f'{list_classes[data_predicted[0]]}', 'FalseNegative'] += 1
        
    df_confusion_matrix.to_csv(batch_output_dir+'ConfusuinMatrix.csv')

def calc_iou(list_true_object, list_pedicted_object):
    
    x1_1 = list_true_object[1]
    y1_1 = list_true_object[2]
    x2_1 = list_true_object[3]
    y2_1 = list_true_object[4]
    
    x1_2 = list_pedicted_object[1]
    y1_2 = list_pedicted_object[2]
    x2_2 = list_pedicted_object[3]
    y2_2 = list_pedicted_object[4]
    
    if x1_1 >= x2_2 or x1_2 >= x2_1 or y1_1 >= y2_2 or y1_2 >= y2_1:
        return 0
    
    x1_overlap = max(x1_1, x1_2)
    y1_overlap = max(y1_1, y1_2)
    x2_overlap = min(x2_1, x2_2)
    y2_overlap = min(y2_1, y2_2)
    
    width = x2_overlap - x1_overlap
    height = y2_overlap - y1_overlap
    area = width * height    
    iou = area / ((x2_1-x1_1)*(y2_1-y1_1))
    
    return iou

if __name__ == '__main__':
    
    ROOT_DIR_DATASET = 'datasets/20231107_test/'
    ROOT_DIR_OUTPUT = 'runs/cross_validation/'
    LIST_CLASSE = ['MV', 'IVS', 'TV'] #labelImgで定義した順
    SPLIT_NUM = 5
    IOU_THRESHOLD = 0.25
    CONF_THRESHOLD = 0.25
    
    for exp in range(100):
        output_dir = ROOT_DIR_OUTPUT + f'exp{exp}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            break
    
    shutil.copytree(ROOT_DIR_DATASET, output_dir+'copy/')
    copied_dataset_dir = output_dir + f"copy/"
    yaml_path = copied_dataset_dir + 'dataset.yaml'
    
    df_info = setting(ROOT_DIR_DATASET, SPLIT_NUM, ROOT_DIR_DATASET)
    df_info = pd.read_csv(ROOT_DIR_DATASET+'InfoDataset.csv')
    gather_data_into_trainval(ROOT_DIR_DATASET, df_info)
    
    for batch_num in range(SPLIT_NUM):
        
        split_data(copied_dataset_dir, batch_num, df_info)
        model_path = train(yaml_path, batch_num, output_dir)
        evaluate(copied_dataset_dir, model_path, IOU_THRESHOLD, LIST_CLASSE, CONF_THRESHOLD, output_dir+f'results/batch{batch_num}/')
        gather_data_into_trainval(copied_dataset_dir, df_info)
        
    shutil.rmtree(copied_dataset_dir)