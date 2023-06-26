# One Row KeyBoard
"0〜9の数字+Spaceのみで表現された数字列"から"その数字列が表す文章"を予測する問題。

PyTorchを用いた深層学習により、元の文章を予測する問題として作成した。

## 入出力
### 入力
0〜9の数字+Spaceのみで表現された文字列。

### 出力
対応する文章

### 例
入力: 87457 91917 418519 5419145 910650710 6441734176 641 9117 50 641 87457 91917 21014

出力: bahia cocoa review showers continued throughout the week in the bahia cocoa zone,

## 使用データ
nltkに収録されているReuters Corpusのうち200データを使用。
なおデータの作成の仕方は、Ground Truthのデータの各文字をASCII文字にした時の下1桁の数字に置き換えていっている。
また、大文字は全て小文字に変換してデータセットを作成した。
### ./data/RowData
処理前の元々のデータ。ここからタブ文字や改行文字の削除等を行ったデータが./data/ConvertDataである。
### ./data/ConvertData
Ground Truthとなるデータ。これがこの問題の出力にあたる。
### ./data/OneRowData
./data/ConvertDataの各文書を数字+spaceのみに変換させたデータ。これがこの問題の入力にあたる。

## 学習
### train.py　+ Model.py
基本的なDNNを用いたコードで学習

### train_gpu.py
GPUを用いてDNNを用いて学習可能。
上記のtrain.pyと使用しているネットワークも異なる。

### OneRowKeyboard.ipynb
.ipynbバージョンの実行ファイル。
train_gpu.pyとネットワークは同じ(ファイルの参照の仕方は異なる。)

問題自体が極めて難しいので、このコードでは精度は高くない。

