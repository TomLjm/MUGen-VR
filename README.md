# MUGen-VR: 统一多模态表征下的视频理解、检索与可控生成系统

**M**ultimodal **U**nified representation for video understanding, retrieval, and controllable **Gen**eration with **V**ideo **R**epresentation

## 项目定位

基于开源 foundation model 构建的一体化系统，支持：
- **理解**：对视频内容进行多模态理解与统一表征
- **检索**：在视频库中进行跨模态检索（文本→视频、图像→视频、音频→视频）
- **生成**：根据多模态条件生成或续写短视频
- **评测**：自动评估生成质量、对齐质量与鲁棒性

## 系统架构

```
                      ┌─────────────────────────┐
                      │   评测与分析层 (Eval)     │
                      │  VBench + 自定义指标      │
                      └──────────┬──────────────┘
                                 │
                      ┌──────────▼──────────────┐
                      │     生成层 (Generation)   │
                      │  generative-models +      │
                      │  LoRA/Adapter + 检索增强   │
                      └──────────┬──────────────┘
                                 │
                      ┌──────────▼──────────────┐
                      │   检索与理解层 (Retrieval)│
                      │  InternVideo + Rerank    │
                      └──────────┬──────────────┘
                                 │
                      ┌──────────▼──────────────┐
                      │ 多模态编码与统一表征层     │
                      │  ImageBind + Unified API │
                      └──────────┬──────────────┘
                                 │
                      ┌──────────▼──────────────┐
                      │    数据与特征层 (Data)    │
                      │  切片 / 特征缓存 / Schema │
                      └─────────────────────────┘
```

## 核心创新点

1. **A：层次化多模态条件融合** — 不同层次特征（低层外观/中层音频/高层语义）动态融合
2. **B：检索增强的视频生成** — Retrieve-then-Generate pipeline
3. **C：模态缺失鲁棒性训练** — Modality Dropout + Confidence Weighting

## 快速开始

```bash
# 创建 conda 环境
conda create -n mugen python=3.10 -y
conda activate mugen
pip install -r requirements/base.txt

# 准备数据
python scripts/prepare_data/prepare_dataset.py --config configs/data/default.yaml

# 提取特征
python scripts/extract_features/extract_all.py --config configs/data/default.yaml

# 运行检索
python scripts/infer/retrieval_demo.py --query "your text query"

# 运行生成
python scripts/infer/generation_demo.py --text "a dog running" --image input.jpg
```

## 目录结构

```
MUGen-VR/
├── src/           # 核心源码（原创模块）
│   ├── data/      # 数据加载与预处理
│   ├── encoders/  # 统一多模态编码器接口
│   ├── fusion/    # 多模态融合模块（创新点A）
│   ├── retrieval/ # 跨模态检索
│   ├── generation/# 条件视频生成（创新点B）
│   ├── evaluation/# 评测与报告
│   ├── serving/   # API与Demo服务
│   └── common/    # 公共接口与工具
├── configs/       # 配置文件
├── scripts/       # 运行脚本
├── experiments/   # 实验结果
├── docs/          # 文档
├── demos/         # 演示应用
└── third_party/   # 第三方基座Repo
```

## 基座依赖

- [InternVideo](https://github.com/OpenGVLab/InternVideo) — 视频编码与检索主干
- [ImageBind](https://github.com/facebookresearch/ImageBind) — 多模态统一表征
- [generative-models](https://github.com/Stability-AI/generative-models) — 视频生成后端
- [VBench](https://github.com/Vchitect/VBench) — 视频生成质量评测
- [CLAP](https://github.com/LAION-AI/CLAP) — 音频增强（可选）

## 阶段路线

| Phase | 目标 | 时间 |
|-------|------|------|
| 1 | Baseline 打通（最小闭环） | Week 1-2 |
| 2 | 统一表征与桥接 | Week 3 |
| 3 | 融合模块与创新点A | Week 4 |
| 4 | 检索增强生成（创新点B） | Week 5 |
| 5 | 鲁棒性训练（创新点C） | Week 6 |
| 6 | 评测闭环与Demo | Week 7-8 |

## License

本项目仅用于学术研究和面试展示。
