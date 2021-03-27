# Saizeriya_detection
サイゼリヤの間違い探しを片面だけで行おうというコードになります．詳細はこちらの記事を参照してください．  

[Coming soon]()

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

