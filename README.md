# One Row KeyBoard
"0〜9の数字+Spaceのみで表現された数字列"から"その数字列が表す文章"を予測する問題。

PyTorchを用いた深層学習により、元の文章を予測する問題として作成した。

## 入出力
### 入力
0〜9の数字+Spaceのみで表現された文字列。

### 出力
対応する文章

### 例
入力: 65235 79795 296397 3419145

出力: BAHIA COCOA REVIEW Showers

## 使用データ
nltkに収録されているReuters Corpusのうち200データを使用。
なおデータの作成の仕方は、Ground Truthのデータの各文字をASCII文字にした時の下1桁の数字に置き換えていっている。
### ./data/RowData
処理前の元々のデータ。ここからタブ文字や改行文字の削除等を行ったデータが./data/ConvertDataである。
### ./data/ConvertData
Ground Truthとなるデータ。これがこの問題の出力にあたる。
### ./data/OneRowData
./data/ConvertDataの各文書を数字+spaceのみに変換させたデータ。これがこの問題の入力にあたる。

