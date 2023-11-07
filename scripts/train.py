from ultralytics import YOLO

if __name__ == '__main__':

     #https://farml1.com/yolov8/

     #公式ドキュメント
     #https://docs.ultralytics.com/usage/cfg/#augmentation

     # Load a model
     model = YOLO("yolov8l.pt")

     #モデル形状に影響するパラメータ:epochs, batch, degrees
     #degree:0.0~指定値間で画像をランダムに回転させるデータ拡張
     model.train(data="datasets/20231001_4chamber/dataset.yaml", epochs=300, batch=2**3, workers=1, degrees=180, pretrained=True)