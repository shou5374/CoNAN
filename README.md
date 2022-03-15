# CoNAN

[CoNAN]()は、語義の関連情報を効率的に抽出し、より明確な語義予測を可能にした、Word Sense Disambiguation(WSD)に対する新たな手法です。複数の語義関連情報の中から、今まで使用されたことが少ない「例文」を採用しました。

# Try the model

## Requirements

- Linux system, e.g. Ubuntu 18.04LTS
- CUDA 11.1
- Anaconda

## Setup

以下のコマンドで環境を作成します。

```
bash bin/setup.sh -c
```

以下のリソースをダウンロードします。

- [Wikipedia Freqs](https://drive.google.com/file/d/1WqNKZZFXM1xrVlDUOFSwMBINJGFlbM_l/view). PMI スコアを計算するために必要なファイルであり、[Barba](https://github.com/SapienzaNLP/consec)らによって共有されたものになります。環境作成後に生成される corpus/フォルダ直下に配置してください。

- (Optional) SemCor で訓練したモデルの[チェックポイント](https://drive.google.com/file/d/1F5f1WNRGVSQ6qZaRgnP3z4p3Wa3ezDod/view?usp=sharing)を共有しています。checkpoints/フォルダを生成し、ダウンロードしたチェックポイントファイルを配置してください。

# Train

以下のコマンドで訓練を実行できます。

```
(conan) user@user-pc:~/conan$ python run.py
```

# Evaluate

以下のコマンドで検証を実行できます。

```
(conan) user@user-pc:~/conan$ python run.py --testing
```

<!-- # Citation
```
@inproceedings{asakawa-2022-conan,
    title = "Knowledge Injection with Constrained Attention Networks for Word Sense Disambiguation",
    author = "Asakawa shou",

}
``` -->

# License

This work is under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
