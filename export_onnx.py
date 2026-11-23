# -*- coding: utf-8 -*-
"""把训练好的 Detector 导出为 ONNX（固定 4 秒输入, 本地 CPU 推理用）。

用法: py export_onnx.py
产出: outputs/detector.onnx
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from train import Detector  # noqa: E402

ROOT = Path(__file__).parent
MAX_LEN = 16000 * 4


def main():
    ckpt = torch.load(ROOT / "outputs" / "best_model.pt",
                      map_location="cpu", weights_only=False)
    model = Detector(str(ROOT / "wavlm-model"))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, MAX_LEN)
    out = ROOT / "outputs" / "detector.onnx"
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["wav"], output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    print(f"已导出 {out}")


if __name__ == "__main__":
    main()
