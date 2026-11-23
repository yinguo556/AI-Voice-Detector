# -*- coding: utf-8 -*-
"""本地推理服务: 上传音频 -> 返回 AI合成/真实人声 判断。

启动: py app.py  (浏览器打开 http://127.0.0.1:8000)
接口: POST /predict  (multipart 上传音频文件, 字段名 file)
"""
import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from scipy.signal import resample_poly

BASE = Path(__file__).parent
MAX_LEN = 16000 * 4
sess = ort.InferenceSession(str(BASE / "outputs" / "detector.onnx"), providers=["CPUExecutionProvider"])

app = FastAPI(title="合成音检测 Demo")


def preprocess(data: bytes) -> np.ndarray:
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
    return wav[None, :].astype(np.float32)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time()
    wav = preprocess(await file.read())
    logits = sess.run(None, {"wav": wav})[0][0]
    prob_fake = float(np.exp(logits[1]) / np.sum(np.exp(logits)))
    return {
        "label": "fake" if prob_fake > 0.5 else "real",
        "label_cn": "AI合成人声" if prob_fake > 0.5 else "真实人声",
        "confidence": round(max(prob_fake, 1 - prob_fake), 4),
        "prob_fake": round(prob_fake, 4),
        "filename": file.filename,
        "inference_ms": round((time.time() - t0) * 1000),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <meta charset="utf-8"><title>合成音检测</title>
    <body style="font-family:sans-serif;max-width:600px;margin:40px auto">
    <h2>合成音 / 真实人声 检测</h2>
    <input type="file" id="f" accept="audio/*">
    <button onclick="go()">检测</button><pre id="r"></pre>
    <script>
    async function go(){
      const fd = new FormData(); fd.append("file", document.getElementById("f").files[0]);
      const t = performance.now();
      const res = await fetch("/predict", {method:"POST", body:fd});
      const j = await res.json();
      document.getElementById("r").textContent = JSON.stringify(j, null, 2);
    }
    </script></body>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
