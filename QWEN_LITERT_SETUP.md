# Qwen 3.5 LiteRT Conversion Setup

This guide provides instructions on how to convert the Qwen 3.5 model (e.g., Qwen3.5-0.8B or Qwen3.5-2B) to LiteRT (`.tflite`) format.

## Prerequisites

1. A machine with sufficient RAM/VRAM to download and process the model (a larger machine or GPU-accelerated instance is recommended for larger models).
2. A Hugging Face account and an access token (`HF_TOKEN`) to download the model.

## Environment Setup

The required Python packages are listed in `requirements_conda.txt`.

1. Create a Python environment (e.g., using `conda`):
   ```bash
   conda create -n qwen_litert python=3.13
   conda activate qwen_litert
   ```

2. Install the necessary dependencies from the provided requirements file:
   ```bash
   pip install -r requirements_conda.txt
   ```
   *Note: Ensure you install the `ai-edge-litert-nightly`, `ai-edge-quantizer-nightly`, and nightly PyTorch builds as configured in the environment.*

## Conversion Process

### 1. Download the Qwen Model
Set your Hugging Face token and download the model checkpoint using the HF Hub.
```python
import os
from huggingface_hub import snapshot_download

# Set your Hugging Face Token (or export it in your shell environment)
os.environ['HF_TOKEN'] = 'your_hf_token'

model_path = snapshot_download(
    'Qwen/Qwen3.5-0.8B', # Or 'Qwen/Qwen3.5-2B'
    token=os.environ['HF_TOKEN'], 
    ignore_patterns=['*.bin','*.msgpack','*.onnx']
)
print(f'Model downloaded to: {model_path}')
```

### 2. Run the Converter Script
Once the model is downloaded and `litert-torch` is properly patched with the Gated DeltaNet attention layers, use the conversion script.

```bash
# Create an output directory
mkdir -p ./output

# Run the Litert PyTorch Converter
python -m litert_torch.generative.examples.qwen.convert_v3_5_to_tflite \
    --checkpoint_path=/path/to/downloaded/model \
    --output_path=./output/ \
    --quantize=dynamic_int8 \
    --kv_cache_max_len=2048
```

## Verify Output
After the conversion script finishes, check the `./output/` directory. You should see `.tflite` files representing the converted model weights and architecture.

## Notes on the Custom Patches
The `/litert-torch` directory contains necessary modifications for Qwen 3.5 conversion:
- **`model_config.py`**: Added `GatedDeltaNetConfig` to support linear attention.
- **`gated_deltanet.py`**: Implementation of `GatedDeltaNetAttention` and `HybridTransformerBlock`.
- **`qwen3_5.py`**: The model definition mapping the hybrid full-attention/linear-attention structure of Qwen3.5.
- **`convert_v3_5_to_tflite.py`**: The entrypoint script to build and convert the model.
- Removed hard imports of `torchao.quantization.pt2e.quantize_pt2e` dynamically to fix TorchAO compatibility issues on newer nightly versions.
