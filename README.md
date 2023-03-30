# fasterrcnn_custom

胸部X線画像から結節を検出するモデル
pytorchのtorchvisionにある[FasterRCNN](https://pytorch.org/vision/main/models/faster_rcnn.html)のモデル構造を書き換えやすいようにファイルを整理したもの

実際にFasterRCNNを改良しており，結節などを検出すると同時に入力された画像にどんな病原体があるかを分類するサブタスクを追加している

## 学習データについて


