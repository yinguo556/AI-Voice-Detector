# -*- coding: utf-8 -*-
"""
从 AISHELL-3 中按说话人抽样真实音频，重采样为 16kHz 单声道，截断至 10 秒，
输出到 dataset/ 目录，并生成 manifest 与伪造合成用的文本清单。

说话人划分（80/20，固定随机种子）：
  - train 说话人 -> 抽 20000 句 -> dataset/train/real/
  - test 说话人  -> 抽 2000 句  -> dataset/test/seen/real/ (1000) + dataset/test/unseen/real/ (1000)

用法: py prepare_real_data.py
"""
import csv
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SRC_ROOT = Path(r"d:\work\aishell3\aishell3")
OUT_ROOT = Path(r"d:\work\dataset")
TARGET_SR = 16000
MAX_DUR_S = 10.0
TRAIN_N = 20000
TEST_N = 2000
SEED = 42

TEXT_TAG_RE = re.compile(r"#\d")


def load_index():
    """扫描所有说话人的 prosody.txt -> {utt_id: (speaker, text)}"""
    index = {}
    for spk_dir in sorted(SRC_ROOT.iterdir()):
        prosody = spk_dir / "prosody" / "prosody.txt"
        if not prosody.exists():
            continue
        with open(prosody, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                utt_id, raw_text = parts[0].strip(), parts[1].strip()
                text = TEXT_TAG_RE.sub("", raw_text)  # 去掉 #1 #3 等韵律标记
                if text:
                    index[utt_id] = (spk_dir.name, text)
    return index


def convert_wav(src: Path, dst: Path):
    """读取 wav -> 单声道 -> 16kHz -> 截断至 MAX_DUR_S -> 写 PCM_16"""
    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != TARGET_SR:
        g = np.gcd(sr, TARGET_SR)
        mono = resample_poly(mono, TARGET_SR // g, sr // g).astype(np.float32)
    max_len = int(MAX_DUR_S * TARGET_SR)
    if len(mono) > max_len:
        mono = mono[:max_len]
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), mono, TARGET_SR, subtype="PCM_16")


def main():
    print("[1/4] 扫描标注文件 ...")
    index = load_index()
    speakers = sorted({spk for spk, _ in index.values()})
    print(f"    说话人 {len(speakers)} 名，标注音频 {len(index)} 条")

    rng = random.Random(SEED)
    shuffled = speakers[:]
    rng.shuffle(shuffled)
    n_test_spk = max(1, round(len(shuffled) * 0.2))
    test_spks = set(shuffled[:n_test_spk])
    train_spks = set(shuffled[n_test_spk:])
    print(f"    训练说话人 {len(train_spks)} 名，测试说话人 {len(test_spks)} 名")

    by_split = {"train": [], "test": []}
    for utt_id, (spk, text) in index.items():
        by_split["test" if spk in test_spks else "train"].append((utt_id, spk, text))
    rng.shuffle(by_split["train"])
    rng.shuffle(by_split["test"])
    sel_train = by_split["train"][:TRAIN_N]
    sel_test = by_split["test"][:TEST_N]
    sel_seen, sel_unseen = sel_test[: TEST_N // 2], sel_test[TEST_N // 2:]
    print(f"    抽样: 训练 {len(sel_train)}，测试 seen {len(sel_seen)}，测试 unseen {len(sel_unseen)}")

    jobs = [
        (sel_train, OUT_ROOT / "train" / "real", "train/real"),
        (sel_seen, OUT_ROOT / "test" / "seen" / "real", "test/seen/real"),
        (sel_unseen, OUT_ROOT / "test" / "unseen" / "real", "test/unseen/real"),
    ]

    print("[2/4] 转换音频 ...")
    rows, text_rows, fails = [], [], 0
    for selection, out_dir, rel in jobs:
        for i, (utt_id, spk, text) in enumerate(selection):
            src = SRC_ROOT / spk / "wav" / f"{utt_id}.wav"
            dst = out_dir / f"{utt_id}.wav"
            try:
                convert_wav(src, dst)
            except Exception as e:
                fails += 1
                print(f"    [失败] {src}: {e}", file=sys.stderr)
                continue
            rows.append((f"dataset/{rel}/{utt_id}.wav", 0, f"aishell3_{spk}"))
            text_rows.append((utt_id, text))
            if (i + 1) % 2000 == 0:
                print(f"    {rel}: {i + 1}/{len(selection)}")

    print("[3/4] 写 manifest 与文本清单 ...")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "manifest_real.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source"])
        w.writerows(rows)
    with open(OUT_ROOT / "real_texts.tsv", "w", encoding="utf-8", newline="") as f:
        for utt_id, text in text_rows:
            f.write(f"{utt_id}\t{text}\n")

    print("[4/4] 统计时长 ...")
    durs = []
    for r in rows[:: max(1, len(rows) // 2000)]:  # 抽 2000 条估时长
        info = sf.info(str(OUT_ROOT.parent / r[0]))
        durs.append(info.duration)
    print(f"    完成: 共 {len(rows)} 条，失败 {fails} 条，"
          f"平均时长 {np.mean(durs):.2f}s，总时长约 {sum(durs) * len(rows) / len(durs) / 3600:.1f} 小时")


if __name__ == "__main__":
    main()
