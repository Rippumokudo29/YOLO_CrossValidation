# YOLO_CrorrValidation

## 使い方
1. 下記構造のデータを用意（通常の学習時と同じですが，画像とラベルは必ず1対1対応でお願いします）
   <br/>datasets/
    -(データセット名)/
       -images/
         -image1.jpg
         -image2.jpg
         -
       -labels/
         -label1.txt
         -label2.t
3. 247~252行目の定数を設定
runs/CrossValidation/exp*/results/batch*/ConfusuinMatrix.csv
