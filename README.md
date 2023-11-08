# YOLO_CrorrValidation

## 使い方
1. 通常の学習時と同じデータセットを用意する．
2. scripts/cross_validation.py内の247~252行目を編集．
+ ROOT_DIR_DATASET:データセットのルートディレクトリ
+ ROOT_DIR_OUTPUT：データの出力先（そのままでよい）
+ LIST_CLASSES：labelImgで定義したクラス名のリスト（labelImgで定義した順で）
+ SPLIT_NUM：交差検証における分割数
+ IOU_THRESHOLD：真陽性と判定するIoUの閾値
+ CONF_THRESHOLD：陽性と判定するYOLO確信度の閾値
  
3. 実行
`python　scripts/cross_validation.py` 
4. {ROOT_DIR_OUTPUT}/exp*/results/batch*/ConfusuinMatrix.csvに各検証結果として混合行列データが出力される．
