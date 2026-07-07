# 多模态视频理解-检索-可控生成项目交接文档

## 1. 项目目标

### 项目名称
MUGen-VR: 统一多模态表征下的视频理解、检索与可控生成系统

### 项目定位
这是一个面向多模态大模型、CV、多模态生成、保研面试与算法实习的研究型工程项目。目标不是搭一个简单 demo，而是基于开源 foundation model 构建一个完整系统，支持以下能力：

- 输入文本、图像、音频中的任意一种或多种条件
- 对视频内容进行理解与统一表征
- 在视频库中进行跨模态检索
- 根据条件生成或续写短视频
- 自动评估生成质量、对齐质量与鲁棒性

### 项目核心卖点
一句话定义：

统一多模态表征下的视频理解、检索与可控生成一体化系统。

面试和保研时的主叙事应围绕以下几点展开：

- 不是单一文生视频，而是多模态理解、检索、生成、评测闭环
- 不是从零训练大模型，而是基于高质量开源底座做系统级创新
- 创新重点在统一接口、模态融合、检索增强生成、鲁棒性训练和评测闭环

## 2. 整体架构蓝图

系统分为五层：

1. 数据与特征层
2. 多模态编码与统一表征层
3. 检索与理解层
4. 生成层
5. 评测与分析层

### 2.1 逻辑数据流

1. 输入文本、图像、音频、视频
2. 用不同编码器抽取模态特征
3. 映射到统一 embedding 空间
4. 做跨模态检索，找到参考视频片段
5. 将文本/图像/音频条件和检索到的参考特征送入生成模块
6. 生成短视频或进行视频续写
7. 使用评测模块自动输出质量、时序一致性、跨模态对齐和失败分析报告

### 2.2 模块划分

- `data/`: 原始数据、切片、metadata、特征缓存
- `encoders/`: 各模态编码器封装
- `fusion/`: 统一表征与条件融合模块
- `retrieval/`: 跨模态检索与 rerank
- `generation/`: 条件视频生成/续写
- `evaluation/`: 评测与失败分析
- `serving/`: demo 与推理服务
- `experiments/`: 实验配置、日志、结果表

## 3. 推荐的 GitHub 基座 Repo

### 3.1 视频理解与检索主干
- Repo: OpenGVLab/InternVideo
- 链接: https://github.com/OpenGVLab/InternVideo
- 作用:
  - 视频 encoder 主干
  - text-video retrieval baseline
  - 可选的视频问答/长视频理解扩展
- 选择原因:
  - 视频基础模型体系完整
  - 与多模态理解强相关
  - 后续可作为生成前 reference retrieval 的核心主干

### 3.2 多模态统一表示原型
- Repo: facebookresearch/ImageBind
- 链接: https://github.com/facebookresearch/ImageBind
- 作用:
  - 图像、文本、音频 embedding 到统一空间
  - 作为跨模态对齐 baseline
- 选择原因:
  - 多模态绑定能力强
  - 适合做统一表征和融合模块的起点
- 注意事项:
  - 许可证为非商业研究友好，适合项目与面试展示，不适合包装成商业产品

### 3.3 视频生成后端
- Repo: Stability-AI/generative-models
- 链接: https://github.com/Stability-AI/generative-models
- 作用:
  - image-to-video 或 text-guided short video generation baseline
  - 后续接入多模态条件和 reference feature
- 选择原因:
  - 视频生成能力成熟
  - 适合做 adapter/LoRA 级微调

### 3.4 视频生成评测
- Repo: Vchitect/VBench
- 链接: https://github.com/Vchitect/VBench
- 作用:
  - 视频生成质量评测
  - 时序一致性和多维指标评估
- 选择原因:
  - 可以直接作为项目里的自动评测基础设施
  - 适合扩展自定义对齐与鲁棒性报告

### 3.5 可选音频增强基座
- Repo: LAION-AI/CLAP
- 链接: https://github.com/LAION-AI/CLAP
- 作用:
  - 增强音频-文本对齐
  - 用于音频条件更强的版本
- 使用时机:
  - 当音频输入成为重点时引入
  - 首版项目可暂不引入，避免初期复杂度过高

## 4. 自建主仓库建议

不要把四个基座 repo 粘在一个目录里，也不要把最终项目描述为“某某项目二次开发”。

建议新建自己的空白主仓库，例如：

`mugen-vr/`

建议目录结构：

```text
mugen-vr/
  README.md
  pyproject.toml
  requirements/
  configs/
    data/
    training/
    inference/
    evaluation/
  docs/
    architecture.md
    decisions/
    reports/
  scripts/
    prepare_data/
    extract_features/
    train/
    infer/
    eval/
  src/
    data/
    encoders/
    fusion/
    retrieval/
    generation/
    evaluation/
    serving/
    common/
  experiments/
  demos/
  third_party/
```

### third_party 组织建议

可用两种策略之一：

1. 使用 git submodule
2. 记录 commit hash，并通过独立环境安装

推荐：
- `third_party/InternVideo`
- `third_party/ImageBind`
- `third_party/generative-models`
- `third_party/VBench`

但核心业务逻辑不写在这些仓库里，而写在 `src/` 中。

## 5. 必须自己写的模块

这些模块是项目真正的“原创部分”，应由主仓库实现，而不是直接拷贝外部 repo。

### 5.1 统一 encoder 接口层
目标：
- 统一不同基座输出格式
- 屏蔽各 repo 差异
- 形成标准化 embedding 接口

接口建议：
- `encode_text()`
- `encode_image()`
- `encode_audio()`
- `encode_video()`

输出统一为：
- embedding tensor
- mask / sequence length
- metadata

### 5.2 多模态融合模块
目标：
- 将文本、图像、音频与视频 reference 特征融合成统一条件
- 形成项目第一核心创新点

建议实现：
- projector
- cross-attention fusion
- gating/confidence weighting
- modality dropout support

建议命名：
- `HierarchicalConditionFusion`

### 5.3 检索增强生成桥接层
目标：
- 将检索到的参考视频特征注入生成模块
- 实现 retrieve-then-generate

建议功能：
- top-k retrieval
- feature aggregation
- reference reranking
- reference-conditioned generation input assembly

### 5.4 评测编排层
目标：
- 串联 VBench、自定义对齐指标和鲁棒性测试
- 输出标准化 report

建议功能：
- generation quality report
- retrieval report
- modality ablation report
- failure clustering report

## 6. 可直接复用但需要封装的模块

### 6.1 InternVideo
可复用：
- 视频 encoder
- retrieval baseline

需要封装：
- embedding 输出统一化
- segment-level retrieval 接口
- rerank hook

### 6.2 ImageBind
可复用：
- text/image/audio embedding

需要封装：
- projector 对齐层
- modality confidence 模块
- 针对视频任务的小规模适配

### 6.3 generative-models
可复用：
- image-to-video / text-to-video baseline

需要改造：
- 增加 reference-conditioned input
- 增加多模态条件注入
- 仅做 adapter/LoRA 级微调，不做全参硬训

### 6.4 VBench
可复用：
- 基础视频质量与时序评测

需要扩展：
- cross-modal alignment metrics
- modality ablation summary
- retrieval-to-generation correlation analysis

## 7. 创新点设计

建议至少落地三个创新点。

### 创新点 A：层次化多模态条件融合
问题：
- 文本、图像、音频对视频生成影响不同
- 不同层次特征需要不同模态信息

做法：
- 构建层次化融合模块
- 低层偏图像外观
- 中层偏音频节奏
- 高层偏文本语义
- 用 gating 动态决定模态贡献

产出：
- 与简单 concat / average baseline 做对比
- 展示在对齐和一致性上的提升

### 创新点 B：检索增强的视频生成
问题：
- 单靠 prompt 的视频生成细节不稳定
- 动作和场景连续性不足

做法：
- 先从视频库中检索相似片段
- 聚合参考视频特征作为生成条件
- 形成 retrieve-then-generate pipeline

产出：
- 与无 reference 生成对比
- 测试生成质量和时序一致性是否提升

### 创新点 C：模态缺失鲁棒性训练
问题：
- 真实输入不一定拥有全部模态
- 模型可能对某一模态过拟合

做法：
- 训练时随机 drop 模态
- 引入 confidence weighting
- 比较单模态、双模态、全模态表现

产出：
- 鲁棒性实验表
- 不同模态组合的性能曲线

## 8. 数据方案

项目第一阶段优先使用公开视频数据，不建议一开始自采。

### 数据要求
- 有视频
- 有文本 caption
- 尽量有音频
- 时长适中，便于切片成短 clip
- 支持公开视频研究使用

### 数据预处理步骤
1. 视频切片
2. 音频抽取
3. 关键帧抽取
4. caption 清洗
5. metadata 标准化
6. 统一样本 ID
7. 特征缓存生成

### 样本标准结构建议
每个 clip 对应：
- `clip_id`
- `video_path`
- `audio_path`
- `keyframes[]`
- `caption`
- `duration`
- `fps`
- `split`
- `cached_features`

## 9. 第一阶段最小闭环

目标：两周内打通一个最小但完整的系统。

### 最小闭环能力
- 文本到视频检索
- 图像/音频到视频相似度计算
- image-to-video 或 text-to-video baseline
- 使用 VBench 跑基础评测

### 验收标准
- 能给定文本检索相关视频
- 能给定图像生成一个短视频 baseline
- 能输出 VBench 评测结果
- 主仓库内有统一脚本完成整个流程

## 10. 迭代路线图

### Phase 1：Baseline 打通
目标：最小闭环

任务：
- 跑通 InternVideo retrieval
- 跑通 ImageBind embedding
- 跑通 generative-models baseline
- 跑通 VBench

产出：
- baseline results
- inference scripts
- 环境说明

### Phase 2：统一表征与桥接
目标：形成主仓库统一接口

任务：
- 封装多模态 encoder API
- 建立特征缓存
- 实现检索统一接口
- 打通 retrieval-to-generation 桥接

产出：
- `src/encoders/`
- `src/retrieval/`
- `src/common/interfaces.py`

### Phase 3：融合模块与创新点 A
目标：做自定义条件融合

任务：
- 写 projector 与 cross-attention fusion
- 加 modality gating
- 实现 modality dropout

产出：
- `src/fusion/`
- 多模态融合实验

### Phase 4：创新点 B 检索增强生成
目标：reference-conditioned generation

任务：
- top-k reference retrieval
- reference aggregation
- 多条件注入生成模型
- baseline 对比实验

产出：
- `src/generation/retrieve_then_generate.py`
- 检索增强实验表

### Phase 5：创新点 C 鲁棒性训练
目标：模态缺失稳定性

任务：
- modality dropout training
- noise robustness experiments
- 模态组合实验

产出：
- ablation tables
- robustness report

### Phase 6：评测闭环与 demo
目标：完整项目包装

任务：
- VBench + 自定义指标统一报告
- failure clustering
- 简单 demo
- 结果展示页

产出：
- `reports/`
- `demos/`
- final results

## 11. 建议的实验矩阵

至少做以下实验：

### 检索实验
- text-to-video retrieval
- image-to-video retrieval
- audio-to-video retrieval
- full multimodal retrieval

指标：
- Recall@1 / 5 / 10
- MRR

### 生成实验
- text-only generation
- image-only generation
- text + image generation
- text + image + retrieved reference generation

指标：
- VBench 总分
- temporal consistency
- motion smoothness
- subject consistency

### 对齐实验
- text-video alignment
- image-video alignment
- audio-video alignment

### 鲁棒性实验
- 去掉文本
- 去掉图像
- 去掉音频
- 文本噪声
- 图像模糊
- 音频扰动

## 12. 算力使用建议

8 张 RTX 3090 的正确使用方式：

### 不推荐
- 全参重训视频生成大模型
- 无约束地扩大数据和分辨率

### 推荐
1. 特征抽取与缓存
2. fusion/projector/reranker 训练
3. 生成模型 LoRA / adapter 微调
4. 多组 ablation 和评测并行

### 资源分配建议
- 2 卡：数据与特征抽取
- 4 卡：检索/融合训练
- 8 卡：生成模型轻量微调与评测

## 13. 建议的 Agent 协作方式

这个项目非常适合多 agent 协作。

### Agent A：项目架构与仓库初始化
职责：
- 新建主仓库
- 建目录结构
- 统一配置与脚本框架
- 搭建 CI / lint / 基础 README

### Agent B：数据与特征工程
职责：
- 数据切片
- metadata 规范
- 关键帧与音频抽取
- 特征缓存流水线

### Agent C：理解与检索模块
职责：
- 接入 InternVideo
- 封装 retrieval API
- 建立 top-k retrieval 与 rerank

### Agent D：统一表征与融合模块
职责：
- 接入 ImageBind
- 写 projector、fusion、gating
- 做 modality dropout

### Agent E：生成模块
职责：
- 接入 generative-models
- 实现 reference-conditioned generation
- 做 LoRA / adapter 微调

### Agent F：评测与分析模块
职责：
- 接入 VBench
- 写 custom metrics
- 写失败案例分析和报告生成

### Agent G：demo 与项目包装
职责：
- 搭 demo
- 输出可视化样例
- 整理最终展示材料

## 14. 与 Claude Code / Codex 协作时的提示词建议

给 agent 的任务不要太大，应切模块拆。

### 合适的任务粒度示例
- 初始化主仓库目录和 Python 包结构
- 为 InternVideo/ImageBind 设计统一 encoder 接口
- 写一个 metadata schema 和数据缓存格式
- 实现 top-k retrieval pipeline
- 实现 HierarchicalConditionFusion 模块
- 把 reference feature 注入生成模型输入
- 写统一 evaluation runner
- 输出 VBench + retrieval + alignment 的 HTML/Markdown 报告

### 不合适的任务粒度
- “帮我做完整项目”
- “帮我训练整个多模态大模型”
- “把这些 repo 拼成能发表的工作”

## 15. 第一批必须完成的任务清单

建议按这个顺序启动。

### Week 1
1. 初始化主仓库
2. 记录四个基座 repo 的 commit
3. 跑通 InternVideo retrieval baseline
4. 跑通 ImageBind embedding demo

### Week 2
5. 跑通 generative-models baseline
6. 跑通 VBench evaluation
7. 写统一数据 schema
8. 写统一 encoder 接口

### Week 3
9. 构建多模态特征缓存
10. 实现 retrieval pipeline
11. 输出第一版 report

### Week 4
12. 写 fusion/gating 模块
13. 做多模态检索实验
14. 完成第一版 retrieve-then-generate

### Week 5-6
15. 做 LoRA/adapter 微调
16. 做 ablation 和鲁棒性实验
17. 完成 demo 和结果展示

## 16. 项目成品的最低交付要求

最终项目至少要有：

- 主仓库源码
- 数据预处理脚本
- 特征缓存脚本
- 训练脚本
- 推理脚本
- 评测脚本
- 统一报告
- demo
- 实验表格
- 失败案例分析

## 17. 推荐的初始实施策略

如果时间和不确定性都要控制，优先走中等难度版本：

### 推荐版本
text / image / audio -> retrieval -> short video generation -> evaluation

### 可降级版本
text / image -> retrieval -> image-to-video generation -> evaluation

### 暂不建议上来就做
- 长视频生成
- 大规模全参联合训练
- 自建超大数据集
- 端到端从头训练统一大模型

## 18. 结论

这个项目的成功关键不是“找到最强的 GitHub repo”，而是：

- 选对底座
- 自己定义统一任务
- 自己写桥接层和融合层
- 自己做 2 到 3 个清晰创新点
- 自己完成评测和失败分析闭环

最终对外叙事应是：

你基于多个 foundation model 设计并实现了一个统一多模态表征下的视频理解、检索与可控生成系统，并围绕模态融合、检索增强生成和鲁棒性训练完成了完整实验闭环。

## 19. 基座 Repo 快速索引

- InternVideo: https://github.com/OpenGVLab/InternVideo
- ImageBind: https://github.com/facebookresearch/ImageBind
- generative-models: https://github.com/Stability-AI/generative-models
- VBench: https://github.com/Vchitect/VBench
- CLAP: https://github.com/LAION-AI/CLAP
