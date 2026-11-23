# -*- coding: utf-8 -*-
"""数据集收尾：去重 train/fake，扫描磁盘生成最终 manifest.csv 并输出统计。"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

DATASET = Path(r"d:\work\dataset")
SPLITS = ["train", "test/seen", "test/unseen"]


def dedupe_train_fake():
    fake_dir = DATASET / "train" / "fake"
    by_utt = defaultdict(list)
    for f in fake_dir.glob("*.wav"):
        by_utt[f.name.split("_v")[0]].append(f)
    removed = 0
    for utt, files in by_utt.items():
        files.sort()
        for extra in files[1:]:
            extra.unlink()
            removed += 1
    return removed, len(by_utt)


def scan():
    rows = []
    for split in SPLITS:
        for label, side in ((0, "real"), (1, "fake")):
            d = DATASET / split / side
            for f in sorted(d.glob("*.wav")):
                src = "aishell3" if side == "real" else f.name.split("_s")[0].split("_v")[1]
                tag = "aishell3" if side == "real" else f"baidu_v{src}"
                rows.append((f"dataset/{split}/{side}/{f.name}", label, tag, split))
    return rows


def main():
    removed, uniq_utts = dedupe_train_fake()
    print(f"去重: 删除 {removed} 条重复伪造音频, train/fake 剩余 {uniq_utts} 条(与 real 一一对应)")

    rows = scan()
    with open(DATASET / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "split"])
        w.writerows(rows)

    print("\n===== 最终数据集构成 =====")
    for split in SPLITS:
        c = Counter(r[1] for r in rows if r[3] == split)
        print(f"{split:12s} 真实 {c[0]:5d}  伪造 {c[1]:5d}  共 {c[0]+c[1]:5d}")
    print(f"{'合计':12s} 真实 {sum(1 for r in rows if r[1]==0):5d}  伪造 {sum(1 for r in rows if r[1]==1):5d}  共 {len(rows):5d}")
    print("\nmanifest.csv 已写入 dataset/manifest.csv (列: path,label,source,split)")


if __name__ == "__main__":
    main()
