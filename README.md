# fasterrcnn_custom

## 胸部X線画像から結節を検出するモデル
pytorchのtorchvisionにある[FasterRCNN](https://pytorch.org/vision/main/models/faster_rcnn.html)

のモデル構造を書き換えやすいようにファイルを整理したもの

実際にFasterRCNNを改良しており，結節などを検出すると同時に入力された画像に結節があるかを分類するサブタスクを追加している

## 学習データについて

訓練・検証データ：[NIHデータセット](https://www.kaggle.com/datasets/nih-chest-xrays/data)

評価データ：[JSRT](http://db.jsrt.or.jp/eng.php)

### ラベルとデータについて
ラベル情報に関しては以下のディレクトリ構造によって管理している
```
dataset
|
├ set1
│ ├ train　.. 訓練データのラベル一覧
│ ├ val　.. 検証データのラベル一覧
│ └ use_class.txt .. 分類するクラス一覧
│
├ test
│ ├ dgree1　.. JSRTの難易度
│ │  ├ train　.. 訓練データのラベル一覧
│ │  ├ val　.. 検証データのラベル一覧
│ │  └ use_class.txt .. 分類するクラス一覧
│ ...
│
│
```
各データのラベル情報はテキストファイルに書かれており

ファイル名を画像ファイル名と一致させてください

テキストファイルのラベル情報の書き方はスペース区切りで

`クラス名 画像の高さ　画像の幅 左上x座標　左上y座標　ボックスの高さ　ボックスの幅`

座標に関しては画像の高さ，幅で０〜1に正規化する

### データ拡張
データ拡張としてcutmix, pixmiを使用しているため

cutmix

画像から病原を切り抜いた画像，貼り付ける範囲を絞るために[肺のセグメンテーションしたマスク画像](https://github.com/IlliaOvcharenko/lung-segmentation)

を用意してもらえると精度がより上げることができます

[pixmix](https://github.com/andyzoujm/pixmix)

リンクのレポジトリーを参考に貼り付ける画像を用意してもらえると使用できます
