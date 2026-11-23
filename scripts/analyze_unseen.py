# -*- coding: utf-8 -*-
import csv
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
from train import compute_metrics

d = np.load("outputs/test_scores.npz")
rows = [r for r in csv.DictReader(open("dataset/manifest.csv", encoding="utf-8"))
        if r["split"] == "test/unseen" and r["label"] == "1"]
assert len(rows) == 851
s_fake, l_fake = d["unseen_scores"][1000:], d["unseen_labels"][1000:]
s_real, l_real = d["unseen_scores"][:1000], d["unseen_labels"][:1000]
for per in ("4", "4105"):
    idx = np.array([r["source"] == f"baidu_v{per}" for r in rows])
    s = np.concatenate([s_real, s_fake[idx]])
    l = np.concatenate([l_real, l_fake[idx]])
    m = compute_metrics(s, l)
    print("留出音色 per=%s (%d条伪造): EER=%.2f%% acc=%.2f%% AUC=%.4f 伪造均分=%.3f"
          % (per, idx.sum(), m["eer"] * 100, m["acc"] * 100, m["auc"], s_fake[idx].mean()))
print("真实样本平均分: %.3f" % s_real.mean())
wrong = [(rows[i]["source"], round(float(s_fake[i]), 3)) for i in range(851) if s_fake[i] <= 0.5]
print("unseen 中误判为真实的伪造:", len(wrong), Counter(w[0] for w in wrong))
