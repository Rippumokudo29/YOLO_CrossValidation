# YOLO_CrorrValidation

## 使い方
1. 通常の学習時と同じデータセットを用意する．
2. scripts/cross_validation内の247~252行目を編集．
+ ROOT_DIR_DATASET:データセットのルートディレクトリ
+ ROOT_DIR_OUTPUT：データの出力先（そのままでよい）
+ LIST_CLASSES：labelImgで定義したクラス名のリスト（labelImgで定義した順で）
+ SPLIT_NUM：交差検証における分割数
+ IOU_THRESHOLD：
