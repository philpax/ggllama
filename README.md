# ggllama

`ggllama` is a Rust port of [ggerganov's llama.cpp](https://github.com/ggerganov/llama.cpp), so that it can be deployed with greater ease.

The current version uses `ggml` directly, so you will require a C compiler. PRs welcome to switch to a more Rust-y solution!

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
