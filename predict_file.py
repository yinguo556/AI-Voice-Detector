# -*- coding: utf-8 -*-
"""命令行快速测试: 对给定音频文件跑推理。

用法: py predict_file.py <音频路径> [更多路径...]
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

BASE = Path(__file__).parent
MAX_LEN = 16000 * 4
sess = ort.InferenceSession(str(BASE / "outputs" / "detector.onnx"), providers=["CPUExecutionProvider"])


def predict(path):
    data = Path(path).read_bytes()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    if len(wav) > MAX_LEN:
        start = (len(wav) - MAX_LEN) // 2
        wav = wav[start: start + MAX_LEN]
    else:
        wav = np.pad(wav, (0, MAX_LEN - len(wav)))
    t0 = time.time()
    logits = sess.run(None, {"wav": wav[None, :].astype(np.float32)})[0][0]
    ms = (time.time() - t0) * 1000
    p = float(np.exp(logits[1]) / np.sum(np.exp(logits)))
    return ("AI合成人声" if p > 0.5 else "真实人声"), p, ms


if __name__ == "__main__":
    for f in sys.argv[1:]:
        label, p, ms = predict(f)
        print(f"{Path(f).name:40s} -> {label}  prob_fake={p:.3f}  ({ms:.0f}ms)")
