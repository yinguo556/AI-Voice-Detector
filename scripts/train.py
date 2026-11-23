# -*- coding: utf-8 -*-
"""
WavLM-Base 微调：合成音(AI) / 真实音频 二分类检测。

训练: dataset/manifest.csv 中 split=train 的样本 (20000 real + 20000 fake)
评估: split=test/seen (见过的说话人+音色), split=test/unseen (留出音色)
指标: EER (主指标), Accuracy, AUC

两段式训练: 前 freeze_epochs 轮冻结 WavLM 主干只训分类头, 之后解冻整体微调。
运行: python train.py  (AutoDL 上需先 export PATH=/root/miniconda3/bin:$PATH)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from soundfile import SoundFile
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, WavLMModel

LABEL_REAL, LABEL_FAKE = 0, 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/root/autodl-tmp/dataset")
    p.add_argument("--model_path", default="/root/autodl-tmp/wavlm-model")
    p.add_argument("--out_dir", default="/root/autodl-tmp/outputs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--freeze_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr_backbone", type=float, default=1e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=16000 * 4, help="4 秒 = 64000 采样点")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    """从 manifest.csv 加载 16kHz 单声道 wav, 定长处理 (训练随机裁剪/测试居中裁剪)。"""

    def __init__(self, rows, data_root, max_len, train=True):
        self.rows = rows
        self.data_root = Path(data_root)
        self.max_len = max_len
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        with SoundFile(str(self.data_root.parent / path)) as f:
            wav = f.read(dtype="float32")
        if len(wav) > self.max_len:
            if self.train:
                start = random.randint(0, len(wav) - self.max_len)
            else:
                start = (len(wav) - self.max_len) // 2
            wav = wav[start: start + self.max_len]
        else:
            wav = np.pad(wav, (0, self.max_len - len(wav)))
        return torch.from_numpy(wav), label


def collate(batch):
    wavs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return wavs, labels


def load_rows(manifest, splits):
    import csv
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] in splits:
                rows.append((r["path"], int(r["label"])))
    return rows


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for wavs, labs in loader:
        logits = model(wavs.to(device))
        scores.extend(F.softmax(logits, dim=1)[:, LABEL_FAKE].cpu().numpy().tolist())
        labels.extend(labs.numpy().tolist())
    scores, labels = np.array(scores), np.array(labels)
    return compute_metrics(scores, labels), scores, labels


def compute_metrics(scores, labels):
    """EER + 准确率(0.5阈值) + AUC"""
    eer, thresh = eer_score(labels, scores)
    acc = accuracy_score(labels, scores > 0.5)
    auc = roc_auc_score(labels, scores)
    return {"eer": round(float(eer), 4), "acc": round(float(acc), 4),
            "auc": round(float(auc), 4), "eer_threshold": round(float(thresh), 4)}


def eer_score(labels, scores):
    """等错误率: 遍历候选阈值使 FPR 与 FNR 最接近, sklearn 实现。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thresholds[idx])


class Detector(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(model_path)
        hidden = self.wavlm.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )

    def forward(self, wavs):
        out = self.wavlm(input_values=wavs).last_hidden_state  # (B, T, H)
        return self.head(out.mean(dim=1))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}, data={args.data_root}, model={args.model_path}")

    train_rows = load_rows(Path(args.data_root) / "manifest.csv", {"train"})
    seen_rows = load_rows(Path(args.data_root) / "manifest.csv", {"test/seen"})
    unseen_rows = load_rows(Path(args.data_root) / "manifest.csv", {"test/unseen"})
    print(f"train {len(train_rows)}, seen {len(seen_rows)}, unseen {len(unseen_rows)}")

    train_ds = AudioDataset(train_rows, args.data_root, args.max_len, train=True)
    seen_ds = AudioDataset(seen_rows, args.data_root, args.max_len, train=False)
    unseen_ds = AudioDataset(unseen_rows, args.data_root, args.max_len, train=False)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate,
                          pin_memory=True, drop_last=True)
    seen_ld = DataLoader(seen_ds, batch_size=args.eval_batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate)
    unseen_ld = DataLoader(unseen_ds, batch_size=args.eval_batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate)

    model = Detector(args.model_path).to(device)
    head_params = list(model.head.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    for p in backbone_params:
        p.requires_grad = False  # 先冻结主干

    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

    history, best_eer = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for p in backbone_params:
                p.requires_grad = True
            print(f"[epoch {epoch}] 解冻 WavLM 主干, lr_backbone={args.lr_backbone}")

        model.train()
        t0, total_loss = time.time(), 0.0
        for wavs, labs in train_ld:
            wavs, labs = wavs.to(device, non_blocking=True), labs.to(device)
            loss = F.cross_entropy(model(wavs), labs, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * len(labs)
        sched.step()

        seen_m, _, _ = evaluate(model, seen_ld, device)
        history.append({"epoch": epoch, "train_loss": round(total_loss / len(train_ds), 4),
                        "seen": seen_m})
        print(f"[epoch {epoch}] loss={total_loss/len(train_ds):.4f} "
              f"seen EER={seen_m['eer']*100:.2f}% acc={seen_m['acc']*100:.2f}% "
              f"({time.time()-t0:.0f}s)")

        if seen_m["eer"] < best_eer:
            best_eer = seen_m["eer"]
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       out_dir / "best_model.pt")

    # 最终用最优权重在两组测试集上评估
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    seen_m, seen_s, seen_l = evaluate(model, seen_ld, device)
    unseen_m, unseen_s, unseen_l = evaluate(model, unseen_ld, device)
    print(f"\n===== 最终结果 (best epoch 权重) =====")
    print(f"seen   : EER={seen_m['eer']*100:.2f}% acc={seen_m['acc']*100:.2f}% AUC={seen_m['auc']:.4f}")
    print(f"unseen : EER={unseen_m['eer']*100:.2f}% acc={unseen_m['acc']*100:.2f}% AUC={unseen_m['auc']:.4f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_seen_eer": best_eer,
                   "final_seen": seen_m, "final_unseen": unseen_m}, f, ensure_ascii=False, indent=2)
    np.savez(out_dir / "test_scores.npz",
             seen_scores=seen_s, seen_labels=seen_l,
             unseen_scores=unseen_s, unseen_labels=unseen_l)
    print(f"已保存: {out_dir}/best_model.pt, metrics.json, test_scores.npz")


if __name__ == "__main__":
    main()
