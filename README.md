![提出用１](https://user-images.githubusercontent.com/70077254/120411565-fbf7e900-c38f-11eb-9502-ccbb6dd9b5c5.PNG)
![提出用2](https://user-images.githubusercontent.com/70077254/120411570-fd291600-c38f-11eb-80d9-7de8a71c22a5.PNG)
![提出用3](https://user-images.githubusercontent.com/70077254/120411572-fdc1ac80-c38f-11eb-93b7-c6c2cbc85abe.PNG)
<h1>vgg16を使用(CNN13層、全結合層3層)</h1>
<h2>肺炎を検知。</h2>
<du>
Kaggle Notebook(旧Kernel)でディープラーニングを実装しました。Pytorchを主に使っていて、他にはnumpy, pandas, matplotlib, torchvision, Orderdict, tqdmをインポートしています。演算装置はGPUです。
データセットは胸部のレントゲン写真です。あらかじめ訓練データと検証データ、テストデータに分かれています。
教師あり学習になります。入力したデータに対してテンソル化して、images, labelsから学習モデルを作ることになります。最終的にNormalとPheumonia(肺炎)の可能性を棒グラフ化します。

ミニバッチ法による学習で20~30分程度で学習が完了します。インポート、ローディング、データ拡張の後に、学習を行います。
Normalを0, Pheumoniaを1としてＤataFrame化とラベリング化します。

vgg16は13層のCNNと3層の全結合層からなるライブラリであり、これを使用してモデル学習を行います。

この後は棒グラフを出力する関数を設定し、データを恣意的に選んで肺炎の可能性を棒グラフで出力します。

成果として、医学的知見がなくとも、容易に可能性を検知できる点だと考えています。
</du>
