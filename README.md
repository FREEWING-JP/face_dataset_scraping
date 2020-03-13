# FACE DATASET SCRAPING
# 概要
どこかのサイトから画像を集めてきて、その顔画像をデータセットとして収集するスクリプト群。おまけとして、それらを利用した超解像度CNNの学習も添えている。

超解像度部分は、自分のPytorch templeteプロジェクトにのっとっているため参考に。

# 1.スクレイピング
構成ファイル：scraping.py

PythonライブラリBeautifulSoup4を利用し画像ファイルを収集する。

とあるwebページのように、メインとなるurlから各archiveページにリンクが貼ってあり、その中に各画像へのリンクが存在している事を想定しているため、このスクリプトで対応できない場合はBeautifulSoupについて調べ拡張を施してほしい。

$ pip install beautifulsoup4

ほか、実行して無いと言われたPythonライブラリは適時pipインストールのこと。

$ python scraping.py url

として実行すると、scrapedディレクトリが生成され中に画像ファイルが収集される。

# 2.酷似画像の選別
構成ファイル：separate.py

opencvの特徴点マッチングを用いて、同一の画像をサブディレクトリ内に生成したseparateディレクトリに隔離する。

scraping.pyによって集めた画像にはスクレイピング対象のサイトによっては同じような画像を含む場合がある。データセットとしてそのような画像は不適切であるため、選別を行う。

$ python separate.py threshold

とすると、thresholdが小さいほど「本当に似ている」画像だけをseparateする。80~100程度が実用的と思われる。

# 3.顔部分の取り出し
構成ファイル：face.py

opencvのアニメ顔の検出器を用いて、厳選されたファイルについて顔検出を行う。

それと同時にリサイズを行い128*128のサイズとする。

$python face.py

とすると、scrapedディレクトリ内の各サブディレクトリ直下の画像ファイルから検出した結果をfaceディレクトリを作成した上でそこに出力する。

# 4.super_resolution
校正ディレクトリ：super_resolution

おまけ。pytorchで64*64の画像を128*128に超解像度するシンプルなCNNを組んで学習したのでそのコードと学習パラメータをおいておく。

dataset.py:学習に必要なファイル。

generate.py:modelとparameterを参照してinputからoutputを生成する。

learn.py:学習プログラム。

model.py:モデルの定義ファイル。

shrink.py:学習データ生成のため128*128画像を64*64相当の解像度に落とすプログラム。

なおexampleの2画像はGANで生成した64*64画像に適用した結果である。