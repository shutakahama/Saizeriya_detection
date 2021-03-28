# Saizeriya_detection
サイゼリヤの間違い探しを片面だけで行うコードになります．詳細はこちらの記事を参照してください．  

[サイゼリヤの間違い探しを片面だけで解く](https://qiita.com/grouse324/items/9f49dafc97b78869476f)

中身はFaster-RCNNを使ったObject detectionです．  

## ファイル構成
```
.  
├── main_detection.py  
├── result: 結果画像を入れるディレクトリ  
└── data  
    ├── img: 学習画像を入れるディレクトリ  
    ├── label: 学習画像のラベルを入れるディレクトリ  
    └── test: テスト画像を入れるディレクトリ 
```

ラベルはPascal VOC形式のxmlファイルを想定しています．  
対応する学習画像とラベルの名前（拡張子以前）は合わせてください．  

## 実行

```
python main_detection.py --num_epochs 20 --batch_size 1 --lr 0.005 --base_path ./
```

## 出力の例
上位10個のBounding Boxがスコアとともに表示されます．  
  
<img src=https://user-images.githubusercontent.com/32294580/112741729-18a82700-8fc3-11eb-8c89-bff5f08b6ac8.png width=400>

## 画像の取得元
[エンターテイメント - サイゼリヤ](https://www.saizeriya.co.jp/entertainment/)
