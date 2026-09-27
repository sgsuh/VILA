# AWQ quantization and TinyChat inference

Uses [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq), which `Dockerfile.dev`
installs at `/opt/llm-awq` together with its CUDA kernels (`awq_inference_engine`).
Set the `TORCH_CUDA_ARCH_LIST` build arg for your GPU (default `8.9`, RTX 40 series).

Run everything inside the dev container:

```bash
# 1. Quantize (outputs to runs/awq/<model name>/)
docker compose run --rm vila scripts/awq/quantize_nvila.sh Efficient-Large-Model/NVILA-Lite-2B

# 2. Chat with the quantized model (empty prompt exits)
docker compose run --rm vila scripts/awq/tinychat_demo.sh \
    Efficient-Large-Model/NVILA-Lite-2B runs/awq/NVILA-Lite-2B demo_images/demo_img.png
```

`quantize_nvila.sh` produces:

| File | Contents |
|---|---|
| `smooth-scale.pt` | SmoothQuant activation scales for the W8A8 vision tower (`smooth_scale.py`) |
| `awq-search.pt` | AWQ scale/clip search results for the LLM |
| `llm-w4-g128-v2.pt` | W4A16 (group size 128) packed LLM weights |

Calibration media defaults to a demo video; pass your own images/videos as extra arguments
to `quantize_nvila.sh`. The AWQ search uses the `mit-han-lab/pile-val-backup` dataset.
