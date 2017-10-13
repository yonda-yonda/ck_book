# 日曜プログラミング：状態空間時系列分析入門

[状態空間時系列分析入門](https://www.kinokuniya.co.jp/f/dsg-01-9784916092922)の各グラフをpythonで再現する。
数式は[カルマンフィルタ Rを使った時系列予測と状態空間モデル](http://www.kyoritsu-pub.co.jp/bookdetail/9784320112537) を参考にする。

## 参考サイト
* http://elsur.jpn.org/ck/
* https://logics-of-blue.com/python-state-space-models/

## イントロ
ナイル川の流量データを[statsmodels](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/datasets/nile/nile.csv)からコピーしoriginal_data配下に置く。
カルマンフィルタの流れと、ライブラリの中で行われている未知数決定の方法を理解する。

## 準備
[原著者のサイト](http://www.ssfpack.com/CKbook.html)からUKdriversKSI.txtをダウンロードし、original_data配下に置く。

## 1章
英国ドライバーの死傷者数を対数変換しプロットする。