# Notice

[llama-rs](https://github.com/setzer22/llama-rs) beat me to the punch. I'll be contributing to that instead.

The original README is preserved below.

---

# ggllama

`ggllama` is a Rust port of [ggerganov's llama.cpp](https://github.com/ggerganov/llama.cpp), so that it can be deployed with greater ease.

The current version uses `ggml` directly, so you will require a C compiler. PRs welcome to switch to a more Rust-y solution!

## Does it work?

Not at the time of writing, no. It runs, but the inference is garbage:

```log
23:59:53 [INFO] ℚ
23:59:54 [INFO]  Насе
23:59:54 [INFO] rsg
23:59:54 [INFO]  eredetiből
23:59:54 [INFO]  Хронологија
23:59:55 [INFO] flug
23:59:55 [INFO]  odkazy
23:59:55 [INFO] orith
23:59:55 [INFO] gior
23:59:56 [INFO]
23:59:56 [INFO]  logs
23:59:56 [INFO] BeanFactory
23:59:56 [INFO] gesamt
23:59:56 [INFO]  bezeichneter
23:59:57 [INFO] Webachiv
23:59:57 [INFO] brie
23:59:57 [INFO]  listade
23:59:57 [INFO] ⊤
23:59:58 [INFO] xtart
23:59:58 [INFO]  kallaste
23:59:58 [INFO] makeText
23:59:58 [INFO]  eredetiből
23:59:59 [INFO] daten
23:59:59 [INFO]  Мос
23:59:59 [INFO] lacht
```

The evaluation returns the wrong logits when given input to process. I'll need to debug this further.

## Build requirements

`ggml-sys` is built with the target features passed into the Rust compiler, so you'll need to set your `RUSTFLAGS` appropriately:

```sh
RUSTFLAGS='-C target-feature=+avx2,+fma,+f16c'
```

Note that `f16c` was stabilised in Rust 1.68.0.

## Model preparation

Model preparation is identical [to the original repo](https://github.com/ggerganov/llama.cpp/blob/master/README.md#usage). This initial version doesn't port `quantize` yet.

I used Conda to create my Python environment:

```sh
conda create --name llama python=3.10
conda activate llama
python3 -m pip install torch numpy sentencepiece

cd vendor/llama.cpp
python3 convert-pth-to-ggml.py models/7B/
```
