"""
quantize_model.py
=================
Quantizes the PyTorch model and ONNX fall detector for ultra-low latency edge devices.
Reduces model footprint by ~50-75% and accelerates CPU / embedded inference.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from run_training import TrainingConfig, FallDetectorResNetLSTM

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_ONNX_QUANT = True
except ImportError:
    HAS_ONNX_QUANT = False

def quantize_pytorch_model(
    checkpoint_path: str = "checkpoints/best_fall_detector.pth",
    output_path: str = "outputs/fall_detector_quantized.pt"
):
    """Dynamic INT8 quantization of linear and LSTM layers in PyTorch."""
    print(f"\n[PyTorch Quantization] Loading checkpoint: {checkpoint_path}")
    cfg = TrainingConfig()
    model = FallDetectorResNetLSTM(
        num_classes=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        num_lstm_layers=cfg.num_lstm_layers,
        bidirectional=cfg.bidirectional,
        dropout_rate=0.0,
        pretrained_backbone=False,
        use_motion_diff=cfg.use_motion_diff
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    orig_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"  Original Checkpoint Size: {orig_size_mb:.1f} MB")

    # Quantize Linear and LSTM layers to 8-bit integers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )

    # Export traced quantized model
    dummy_input = torch.randn(1, 16, 3, 224, 224)
    traced_quant = torch.jit.trace(quantized_model, dummy_input)
    traced_quant.save(output_path)
    quant_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"  ✓ Quantized TorchScript Model Saved: {output_path} ({quant_size_mb:.1f} MB)")
    print(f"  Compression Ratio: {orig_size_mb / quant_size_mb:.2f}x smaller")
    return output_path

def quantize_onnx_model(
    onnx_input: str = "outputs/fall_detector.onnx",
    onnx_output: str = "outputs/fall_detector_quantized.onnx"
):
    """Dynamic INT8 quantization of ONNX model."""
    if not HAS_ONNX_QUANT:
        print("[ONNX Quantization] onnxruntime quantization module not available. Skipping ONNX quant.")
        return None

    print(f"\n[ONNX Quantization] Quantizing {onnx_input} ...")
    orig_size_mb = os.path.getsize(onnx_input) / (1024 * 1024)

    quantize_dynamic(
        model_input=onnx_input,
        model_output=onnx_output,
        weight_type=QuantType.QUInt8
    )

    quant_size_mb = os.path.getsize(onnx_output) / (1024 * 1024)
    print(f"  ✓ Quantized ONNX Model Saved: {onnx_output} ({quant_size_mb:.1f} MB)")
    print(f"  Compression Ratio: {orig_size_mb / quant_size_mb:.2f}x smaller")
    return onnx_output

if __name__ == "__main__":
    pt_quant = quantize_pytorch_model()
    onnx_quant = quantize_onnx_model()
