# -*- coding: utf-8 -*-
"""
百度智能云短文本在线合成：批量为 dataset 中的真实样本合成伪造音频。

配额方案（与 项目方案.md 一致）：
  - 训练集 20000 条：臻品音库 3 音色约 9000 条 + 基础音库 4 音色约 11000 条
  - test/seen 1000 条：用训练用过的音色合成
  - test/unseen 1000 条：度灵儿(4105, 臻品) / 度丫丫(4, 基础) 各 500 条，不进训练

并发：臻品音库限 2，基础音库文档限 100，这里保守用 6。
支持断点续跑：已存在的输出文件自动跳过。
计费：每条文本 <=120 GBK 字节 = 1 次调用（本数据集文本均很短，按 1 次计）。

用法: py synthesize_fake.py
"""
import csv
import json
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

API_KEY = os.environ.get("BAIDU_API_KEY", "")          # 百度智能云 API Key
SECRET_KEY = os.environ.get("BAIDU_SECRET_KEY", "")    # 百度智能云 Secret Key
if not API_KEY or not SECRET_KEY:
    raise SystemExit("请先设置环境变量 BAIDU_API_KEY / BAIDU_SECRET_KEY（勿将密钥写入代码）")
CUID = " coursework-detect-5600"

DATASET = Path(r"d:\work\dataset")
TOKEN_CACHE = DATASET / "baidu_token.json"

# 训练用音色（per -> 档位）；注意 per=4(度丫丫) 留给 test/unseen，不进训练
VOICES_ZHENPIN = {4003: "度逍遥", 4106: "度博文", 4119: "度小鹿"}
VOICES_JICHU = {0: "度小美", 1: "度小宇", 3: "度逍遥"}
# 留出音色：只用于 test/unseen
VOICE_UNSEEN = [(4105, "zhenpin", "度灵儿"), (4, "jichu", "度丫丫")]

WORKERS = {"zhenpin": 2, "jichu": 6}
MAX_RETRY = 4


def get_token():
    if TOKEN_CACHE.exists():
        cache = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
        if cache.get("expires_at", 0) > time.time() + 600:
            return cache["access_token"]
    url = ("https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"
           f"&client_id={API_KEY}&client_secret={SECRET_KEY}")
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    token = data["access_token"]
    TOKEN_CACHE.write_text(json.dumps({
        "access_token": token,
        "expires_at": time.time() + data.get("expires_in", 2592000),
    }), encoding="utf-8")
    return token


def synthesize(token, text, per, spd, pit):
    """返回 wav 二进制；失败抛出异常（带 err_msg）。"""
    tex = urllib.parse.quote(urllib.parse.quote(text))  # 二次 urlencode
    body = ("tex={}&lan=zh&cuid={}&ctp=1&aue=6&vol=5&spd={}&pit={}&per={}"
            "&tok={}&audio_ctrl={}").format(
        tex, urllib.parse.quote(CUID.strip()), spd, pit, per, token,
        urllib.parse.quote('{"sampling_rate":16000}'))
    req = urllib.request.Request(
        "https://tsn.baidu.com/text2audio",
        data=body.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        ctype = r.headers.get("Content-Type", "")
        payload = r.read()
    if ctype.startswith("audio"):
        return payload
    err = json.loads(payload.decode("utf-8", errors="replace"))
    raise RuntimeError(f"err_no={err.get('err_no')} err_msg={err.get('err_msg')}")


_token_lock = threading.Lock()
_token = None


def token():
    global _token
    with _token_lock:
        if _token is None:
            _token = get_token()
        return _token


def refresh_token():
    global _token
    with _token_lock:
        TOKEN_CACHE.unlink(missing_ok=True)
        _token = get_token()


GBK_BYTES = 120


def build_jobs():
    """返回 (utt_id, text, tier, per, out_path) 列表。"""
    texts = {}
    with open(DATASET / "real_texts.tsv", encoding="utf-8") as f:
        for line in f:
            utt_id, text = line.rstrip("\n").split("\t", 1)
            texts[utt_id] = text

    def locate(utt_id):
        for split in ("train", "test/seen", "test/unseen"):
            if (DATASET / split / "real" / f"{utt_id}.wav").exists():
                return split
        return None

    rng = random.Random(42)
    train_utts, seen_utts, unseen_utts = [], [], []
    for utt_id in texts:
        split = locate(utt_id)
        if split == "train":
            train_utts.append(utt_id)
        elif split == "test/seen":
            seen_utts.append(utt_id)
        elif split == "test/unseen":
            unseen_utts.append(utt_id)

    jobs = []
    zhenpin_pers, jichu_pers = list(VOICES_ZHENPIN), list(VOICES_JICHU)
    for utt_id in train_utts:
        # 9000 臻品 : 11000 基础
        if rng.random() < 0.45:
            tier, per = "zhenpin", rng.choice(zhenpin_pers)
        else:
            tier, per = "jichu", rng.choice(jichu_pers)
        spd, pit = rng.randint(3, 7), rng.randint(3, 7)
        out = DATASET / "train" / "fake" / f"{utt_id}_v{per}_s{spd}p{pit}.wav"
        jobs.append((utt_id, texts[utt_id], tier, per, spd, pit, out))
    for utt_id in seen_utts:
        if rng.random() < 0.45:
            tier, per = "zhenpin", rng.choice(zhenpin_pers)
        else:
            tier, per = "jichu", rng.choice(jichu_pers)
        spd, pit = rng.randint(3, 7), rng.randint(3, 7)
        out = DATASET / "test" / "seen" / "fake" / f"{utt_id}_v{per}_s{spd}p{pit}.wav"
        jobs.append((utt_id, texts[utt_id], tier, per, spd, pit, out))
    for i, utt_id in enumerate(unseen_utts):
        per, tier, _name = VOICE_UNSEEN[i % 2]  # 两个留出音色交替，各 500
        spd, pit = rng.randint(3, 7), rng.randint(3, 7)
        out = DATASET / "test" / "unseen" / "fake" / f"{utt_id}_v{per}_s{spd}p{pit}.wav"
        jobs.append((utt_id, texts[utt_id], tier, per, spd, pit, out))
    return jobs


def run_job(job):
    utt_id, text, tier, per, spd, pit, out = job
    if out.exists():  # 断点续跑
        return (job, "skip")
    out.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            wav = synthesize(token(), text, per, spd, pit)
            out.write_bytes(wav)
            return (job, "ok")
        except Exception as e:
            last_err = e
            if "err_no=111" in str(e) or "token" in str(e).lower():
                refresh_token()
            time.sleep(2 ** attempt)
    return (job, f"fail: {last_err}")


def main():
    jobs = build_jobs()
    todo = [j for j in jobs if not j[6].exists()]
    n_zhenpin = sum(1 for j in todo if j[2] == "zhenpin")
    print(f"任务总数 {len(jobs)}，待合成 {len(todo)}（臻品 {n_zhenpin} 次，基础 {len(todo) - n_zhenpin} 次）")
    print("臻品超出 10000 免费额度部分将按量计费；基础约 2 元/千次。")
    input("确认额度无误后按回车开始 ...")

    rows, fails = [], []
    done = 0
    for tier in ("zhenpin", "jichu"):
        tier_jobs = [j for j in todo if j[2] == tier]
        with ThreadPoolExecutor(max_workers=WORKERS[tier]) as pool:
            for job, status in pool.map(run_job, tier_jobs):
                done += 1
                utt_id, _text, t, per, spd, pit, out = job
                if status == "ok" or status == "skip":
                    split = "train" if "train" in out.parts else ("test/unseen" if "unseen" in out.parts else "test/seen")
                    rows.append((f"dataset/{split}/fake/{out.name}", 1,
                                 f"baidu_{t}_per{per}_s{spd}p{pit}"))
                else:
                    fails.append((job[0], job[1], status))
                if done % 500 == 0:
                    print(f"  进度 {done}/{len(todo)}，失败 {len(fails)}")

    with open(DATASET / "manifest_fake.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source"])
        w.writerows(rows)
    if fails:
        with open(DATASET / "synthesize_failed.tsv", "w", encoding="utf-8") as f:
            for utt_id, text, status in fails:
                f.write(f"{utt_id}\t{text}\t{status}\n")
    print(f"完成: 成功/跳过 {len(rows)}，失败 {len(fails)}，manifest 写入 dataset/manifest_fake.csv")


if __name__ == "__main__":
    main()
