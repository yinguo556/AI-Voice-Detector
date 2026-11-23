# 合成音检测系统（AI Voice Detection）

判断一段带人声的音频是 **AI 合成人声** 还是 **真实人声** 的二分类检测系统。课程作业项目。

## 快速开始

```bash
pip install -r requirements.txt
python app.py        # 浏览器打开 http://127.0.0.1:8000, 上传音频即可检测
```

预训练主干与微调权重获取见下文「模型与数据获取」，完整复现流程见「复现 / 运行」。

## 结果速览

| 测试集 | EER | 准确率 | AUC |
|---|---|---|---|
| seen（训练见过的说话人 + 音色） | **0.00%** | 100.00% | 1.0000 |
| unseen（完全留出的说话人 + 音色） | **7.29%** | 79.52%* | 0.9724 |

\* unseen 准确率在 0.5 默认阈值下；EER 是更公平的指标。分组细看：**臻品档留出音色（度灵儿 4105）EER 0.00% 全检出，基础档留出音色（度丫丫 4）EER 11.20%**——模型学到的是音库档位/合成引擎的家族伪影，跨引擎泛化是该领域的核心难点（详见下文「关键发现」）。

本地 CPU（锐龙 R7 5600）推理速度约 **300ms/条**。

## 系统架构

```
数据准备(本地CPU)                训练(AutoDL GPU)              部署(本地CPU)
┌──────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ AISHELL-3 真实音频 │      │ WavLM-Base 两段式微调 │      │ ONNX (detector.onnx) │
│ + 百度TTS伪造音频  │ ──→  │ + MLP 分类头         │ ──→  │ + FastAPI 上传接口   │
│ 43,851 条 16k wav │      │ 10 epochs, RTX4080S │      │ 浏览器上传即出结果    │
└──────────────────┘      └─────────────────────┘      └─────────────────────┘
```

## 数据集构建（本项目自己的工作）

真实音频来自开源中文语料 **AISHELL-3**（218 说话人 / 85 小时 / 88,035 句），伪造音频全部通过 **百度智能云短文本在线合成 API** 生成（本机无显卡，无法本地跑开源 TTS，且 API 合成的伪影分布更贴近真实应用场景）。

- **抽样原则**：按说话人划分训练/测试（174 人训练、44 人测试，种子 42 可复现），同一说话人绝不跨集合，防止模型"认人"而非"认真伪"。
- **配对原则**：每条伪造音频用对应真实样本的文本合成，正负样本仅差声学伪影、文本分布一致。
- **多样性来源**：百度 3 档音库 × 多发音人 × 语速/音调随机扰动（spd/pit 3~7）。训练覆盖 6 个音色（基础：度小美0/度小宇1/度逍遥3；臻品：度逍遥4003/度博文4106/度小鹿4119）。
- **留出设计（泛化实验核心）**：基础档「度丫丫(4)」与臻品档「度灵儿(4105)」两个音色**完全不进训练集**，只用于 unseen 测试，模拟"检测没见过的新一代合成器"。
- 全部音频统一为 **16kHz / 单声道 / wav，截断至 10 秒**。

| 集合 | 真实 | 伪造 | 用途 |
|---|---|---|---|
| train | 20,000 | 20,000 | 严格 1:1 配对 |
| test/seen | 1,000 | 1,000 | 见过的说话人 + 音色 |
| test/unseen | 1,000 | 851 | 留出的说话人 + 音色（缺 149 条为 API 额度耗尽） |

## 方法

- **特征/主干**：微软 **WavLM-Base**（94.6M 参数，自监督预训练语音模型），利用其海量真实语音先验捕捉伪造音频的细微伪影（相位抖动、声码器痕迹）。
- **分类头**：mean-pool → MLP(768→256→2)，Dropout 0.3，label smoothing 0.1。
- **两段式训练**：前 2 轮冻结主干只训分类头（lr 1e-3），之后解冻整体微调（主干 lr 1e-4，batch 32，10 epochs，AdamW + StepLR，梯度裁剪 5.0）。
- **部署**：权重导出 ONNX（opset 18，外部权重 `detector.onnx.data`），`onnxruntime` CPU 推理，FastAPI 提供 `POST /predict` 与简单网页。

## 关键发现（报告核心分析）

1. **同分布测试满分**：seen 测试集 EER 0.00%，说明任务本身在当前数据规模下对 WavLM 微调是容易的。
2. **泛化落差全部来自基础音库留出音色**：unseen 的 379 个错误全部是把度丫丫(4)的伪造判成真人；而质量最高、最拟真的臻品档留出音色（度灵儿）反而 **EER 0% 一条不漏**。
3. **解释**：模型学到的是**合成引擎家族伪影**而非具体音色——臻品各音色共用同一套新一代合成后端，伪影高度同源，跨音色泛化完美；基础档音色间后端差异大，未见过的音色伪影特征漂移明显。这验证了音频深伪检测领域的共识：检测器对未知合成系统的泛化能力是主要瓶颈，训练数据应尽可能覆盖更多独立合成系统。
4. **分数校准偏移**：unseen 上准确率(79.5%)与 EER(7.29%)差距大，AUC 却高达 0.972——排序能力强但绝对分数偏移（EER 阈值 0.047 vs 默认 0.5），实际部署可考虑分档校准阈值。

## 目录结构

> 注：`dataset/`（4.3GB）与 `wavlm-model/`（预训练主干）体积过大不入库，见下文「模型与数据获取」。

```
.
├── README.md                 # 本文件
├── requirements.txt          # Python 依赖清单
├── scripts/
│   ├── download_dataset.py   # 数据集下载 (ModelScope 加载 AISHELL-3)
│   ├── prepare_real_data.py  # 阶段一: AISHELL-3 抽样转码 (16k mono, 按说话人划分)
│   ├── synthesize_fake.py    # 阶段一: 百度TTS批量合成 (音色配额/扰动/留出设计/断点续跑)
│   ├── finalize_dataset.py   # 数据集收尾: 去重 + 生成最终 manifest
│   ├── train.py              # 阶段二: AutoDL 训练 (WavLM微调 + EER评估)
│   └── analyze_unseen.py     # 阶段三: 按音色拆分 unseen 误差分析
├── export_onnx.py            # ONNX 导出
├── predict_file.py           # 命令行推理: python predict_file.py <音频路径>
├── app.py                    # 推理服务: python app.py → http://127.0.0.1:8000
└── outputs/                  # 训练产出 (metrics.json / test_scores.npz / train.log 入库,
                              # 模型权重见下方「模型与数据获取」)
```

## 模型与数据获取

- **WavLM-Base 主干**：请自行下载 Hugging Face [`microsoft/wavlm-base`](https://huggingface.co/microsoft/wavlm-base)（或 [ModelScope 镜像](https://modelscope.cn) 搜索 wavlm-base），放入项目根目录 `wavlm-model/` 即可，`train.py` / `export_onnx.py` 从该目录加载；
- **微调权重**（`best_model.pt`，361MB，超过 GitHub 单文件限制）：**下载请见 [Releases](../../releases)**；复现者按下方流程训练后也可在 `outputs/` 得到等价权重，或运行 `export_onnx.py` 自行导出 `detector.onnx`；
- **数据集**：不入库，按 `scripts/` 中脚本顺序执行即可复现（真实侧来自 AISHELL-3，伪造侧需自备百度智能云 TTS API 额度与密钥环境变量）。

## 复现 / 运行

```bash
pip install -r requirements.txt

# 1. 本地演示（CPU，约300ms/条）
python app.py                     # 浏览器打开 http://127.0.0.1:8000 上传音频
python predict_file.py xxx.wav    # 或命令行直接测

# 2. 重新训练（AutoDL，单卡约1~1.5小时）
#    数据与 wavlm-model 放 /root/autodl-tmp/ 后:
python scripts/train.py

# 阶段一(复现数据集)合成伪造音频前需设置百度API密钥:
# Linux: export BAIDU_API_KEY=xxx BAIDU_SECRET_KEY=xxx
# Windows: set BAIDU_API_KEY=xxx && set BAIDU_SECRET_KEY=xxx
python scripts/synthesize_fake.py

# 3. 重新导出 ONNX (Windows 控制台需 UTF-8)
set PYTHONUTF8=1 && python export_onnx.py
```

## 环境

| 环境 | 配置 |
|---|---|
| 数据准备/部署 | Windows, Python 3.12, numpy/soundfile/scipy/onnxruntime/fastapi |
| 训练 | AutoDL RTX 4080 SUPER, PyTorch 2.8+cu128, transformers 5.24 |
| 已知坑 | torch 2.15 需新版 VC++ 运行库；ONNX 导出需 onnxscript 包；Windows GBK 控制台跑导出需 `PYTHONUTF8=1` |

## 局限与展望

- 伪造数据全部来自百度一家服务商（受无显卡约束），对其它厂商/开源 TTS（ChatTTS、GPT-SoVITS 等）的泛化未验证——后续可接入多家 API 扩充伪造源多样性；
- unseen 中基础档音色检出率偏低，可通过增加基础音库音色数量（买额度合成）改善；
- 未测试对抗场景：重压缩、加噪、变调后的鲁棒性；
- 当前只判断整段真伪，未做伪造片段定位（partial fake）。

## 费用记录

百度 TTS 合成约 2.2 万次调用（臻品免费额度 1 万次 + 基础约 24 元），AutoDL 训练约 1 卡时（个位数元）。总成本 30 元以内。
