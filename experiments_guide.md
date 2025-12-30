# UAV-YOLO 实验框架指导文档

本文档为 "保姆级" 指南，旨在帮助您轻松运行本项目包含的所有对比实验和消融实验，并对结果进行分析。

## 1. 实验框架概述

本框架旨在通过一系列结构化的实验，科学地评估和验证模型改进的有效性。主要包含以下几类实验：

*   **基线实验 (Baseline):** 验证标准YOLO模型在您的数据集上的性能表现，作为后续所有改进的对比基准。
*   **完整模型实验 (Robust Model):** 验证集成了所有鲁棒性模块（去恶劣天气、背景抑制等）的最终模型的性能。
*   **消融实验 (Ablation Studies):** 通过“控制变量法”，逐一验证每个独立模块（如数据增强、频域增强等）对模型整体性能的贡献。
*   **轻量化实验 (Lightweight Model):** 探索在保持鲁棒性的前提下，通过使用更轻量的主干网络来降低模型复杂度和计算量，以适应无人机部署。

## 2. 实验准备

*   **环境:** 请确保您已经根据主 `README.md` 文件配置好了所有必需的依赖环境。
*   **数据:** 确保您的数据集已经按 `uav_dataset.yaml` 或 `configs/base.yaml` 中指定的路径和格式准备就绪。
*   **设备:** 所有训练脚本默认使用 `cuda` 设备。如果您没有兼容的NVIDIA GPU，请手动打开每个 `run_*.sh` 脚本，并将 `--device cuda` 修改为 `--device cpu`。

## 3. 如何运行训练实验

我们为您准备了一系列一键式训练脚本，存放于 `scripts/` 目录下。您只需在终端中运行相应的脚本即可启动训练。

例如，要运行基线模型实验：

```bash
bash scripts/run_baseline.sh
```

训练启动后，所有输出（包括日志和模型权重文件）都会被保存在 `checkpoints/` 目录下，并以脚本中 `--experiment` 参数指定的名称创建唯一的子目录。例如，上述命令的结果会保存在 `checkpoints/baseline/` 中。

---
 
### 实验脚本列表

| 脚本文件                                      | 对应实验                 | 配置文件                               | 实验结果目录 (在 `checkpoints/` 下) |
| --------------------------------------------- | ------------------------ | -------------------------------------- | --------------------------------- |
| `scripts/run_baseline.sh`                     | 基线模型                 | `configs/baseline.yaml`                | `baseline`                        |
| `scripts/run_robust.sh`                       | 完整鲁棒模型             | `configs/robust.yaml`                  | `robust_full`                     |
| `scripts/run_lightweight.sh`                  | 轻量化鲁棒模型           | `configs/lightweight.yaml`             | `lightweight_robust`              |
| `scripts/run_ablation_weather_aug.sh`         | 消融: 仅天气数据增强     | `configs/ablation_weather_aug_only.yaml` | `ablation_weather_aug`            |
| `scripts/run_ablation_bg_suppression.sh`      | 消融: 仅背景抑制模块     | `configs/ablation_bg_suppression.yaml` | `ablation_bg_suppression`         |
| `scripts/run_ablation_freq_enhancement.sh`    | 消融: 仅频域增强模块     | `configs/ablation_freq_enhancement.yaml` | `ablation_freq_enhancement`       |
| `scripts/run_ablation_condition_aware.sh`     | 消融: 仅环境感知模块     | `configs/ablation_condition_aware.yaml`  | `ablation_condition_aware`        |
| `scripts/run_comparison.sh`                   | SOTA对比: YOLOv8n        | `configs/comparison.yaml`             | `runs/detect/train*`              |

---

### 3.1. 一键式SOTA对比实验 (YOLOv8)

为了简化与SOTA模型的对比流程，我们特别集成了`YOLOv8`的训练脚本，让您可以像运行其他实验一样，通过一条命令来训练一个标准的YOLOv8n模型。

| 脚本文件 | 对应实验 | 配置文件 | 实验结果目录 |
|---|---|---|---|
| `scripts/run_comparison.sh` | YOLOv8n 对比模型 | `configs/comparison.yaml` | `runs/detect/train*` |

**运行脚本:**

```bash
bash scripts/run_comparison.sh
```

**重要提示:**

*   该脚本会使用 `ultralytics` 库来训练 `YOLOv8n` 模型。
*   在运行此脚本前，请确保您已经根据项目根目录下的 `requirements.txt` 文件安装了所有依赖，特别是 `ultralytics`。您可以通过以下命令安装：
    ```bash
    pip install -r requirements.txt
    ```
*   训练结果不会保存在 `checkpoints/` 目录下，而是遵循 `ultralytics` 的默认行为，保存在项目根目录下的 `runs/detect/` 目录中（例如 `runs/detect/train`，`runs/detect/train2` 等）。
*   训练完成后，最佳模型会被保存为 `best.pt` 在该结果目录的 `weights/` 子目录下 (例如 `runs/detect/train/weights/best.pt`)。

**评估YOLOv8模型:**

YOLOv8的评估也使用其自身的命令行工具。您可以使用 `yolo` 命令来执行评估。

**评估示例:**

假设训练结果保存在 `runs/detect/train/` 中，您可以使用以下命令进行评估：

```bash
# data参数指向我们项目的数据集配置文件
# model参数指向训练好的YOLOv8模型权重
yolo detect val data=uav_dataset.yaml model=runs/detect/train/weights/best.pt
```

这个评估命令会输出mAP等指标。您需要将这些指标记录到下面的“实验结果总览表”中，与我们自己的模型进行对比。

---

## 4. 如何评估模型性能

训练完成后，您会在每个实验的结果目录中找到一个 `best_model.pth` 文件，这是该次实验中验证集上表现最好的模型权重。

您可以使用 `scripts/eval_on_testset.sh` 脚本来评估任意模型在测试集上的性能。

**使用方法:**

```bash
bash scripts/eval_on_testset.sh <模型权重路径> [配置文件路径] [输出json文件路径]
```

**参数说明:**

*   `<模型权重路径>`: **必需参数**。指向您想要评估的 `.pth` 文件。例如 `checkpoints/baseline/best_model.pth`。
*   `[配置文件路径]`: 可选参数。评估时使用的配置文件，**强烈建议使用与训练时相同的配置文件**，以确保模型结构一致。默认为 `configs/base.yaml`。
*   `[输出json文件路径]`: 可选参数。保存评估结果（如mAP等指标）的 `json` 文件路径。默认为 `results/eval_results.json`。

**评估示例:**

评估**基线模型**:

```bash
bash scripts/eval_on_testset.sh checkpoints/baseline/best_model.pth configs/baseline.yaml results/eval_baseline.json
```

评估**完整鲁棒模型**:

```bash
bash scripts/eval_on_testset.sh checkpoints/robust_full/best_model.pth configs/robust.yaml results/eval_robust_full.json
```

评估**轻量化模型**:

```bash
bash scripts/eval_on_testset.sh checkpoints/lightweight_robust/best_model.pth configs/lightweight.yaml results/eval_lightweight.json
```

**注意:** 您需要为您的**正常天气测试集**和**极端天气测试集**准备不同的数据配置文件（或在 `configs/` 文件中修改路径），并分别运行评估脚本，以获得模型在两种条件下的性能数据。

## 5. 实验结果汇总与分析

为了方便您撰写论文，建议使用以下表格来汇总所有实验结果。

| 实验名称 (Experiment)          | 模型参数量 (M) | mAP@0.5 (正常天气) | mAP@0.5 (极端天气) | FPS (可选用) | 备注 (Notes)                                   |
| ------------------------------ | -------------- | ---------------- | ---------------- | ------------ | ---------------------------------------------- |
| **Baseline**                   | *请填写*       | *请填写*         | *请填写*         | *请填写*     | 标准模型性能                                   |
| **+ Weather Aug Only**         | *请填写*       | *请填写*         | *请填写*         | *请填写*     | 仅数据增强的贡献                               |
| **+ BG Suppression Only**      | *请填写*       | *请填写*         | *请填写*         | *请填写*     | 仅背景抑制的贡献                               |
| **+ Freq Enhancement Only**    | *请填写*       | *请填写*         | *请填写*         | *请填写*     | 仅频域增强的贡献                               |
| **+ Condition Aware Only**     | *请填写*       | *请填写*         | *请填写*         | *请填写*     | 仅环境感知模块的贡献                           |
| **Robust Model (Full)**        | *请填写*       | *请填写*         | *请填写*         | *请填写*     | **我们提出的完整模型**                         |
| **Lightweight Robust Model**   | *请填写*       | *请填写*         | *请填写*         | *请填写*     | 探索轻量化的可能性                             |

**如何填写表格:**

1.  **模型参数量:** 训练开始时，脚本会在终端打印出模型参数量，请记录下来。
2.  **mAP@0.5:** 运行评估脚本后，打开生成的 `.json` 文件，找到mAP指标并填写。您需要为两种天气条件分别评估和填写。
3.  **FPS:** 模型推理速度。这通常需要一个单独的基准测试脚本来测量。

通过对比这张表，您可以清晰地看到每个模块的增益，以及您的最终模型相较于基线模型的提升，这将成为您论文中实验部分的核心论据。

## 6. 与SOTA模型的对比实验指南

为了证明我们模型的先进性，与公认的SOTA（State-of-the-Art）模型进行对比至关重要。这部分实验需要您在本地安装并运行这些外部模型的官方代码库。

**核心思路:** 在**完全相同的数据集**和**测试条件**下，分别训练和评估我们的模型以及SOTA模型，然后比较核心指标。

#### 推荐对比模型

*   **YOLOv8/YOLOv9:** 来自Ultralytics的官方实现，是目前工业界和学术界最常用的基准之一。
*   **YOLOv10:** 来自清华大学的最新研究，以其高效的设计著称。

#### 对比实验分步指南

1.  **准备外部模型环境:**
    *   根据您选择的SOTA模型（例如YOLOv8），访问其GitHub官方仓库。
    *   按照其官方文档说明，在您的电脑上创建一个**新的、独立的环境**来安装其所有依赖。**（请勿在当前项目环境中安装，避免依赖冲突）**

2.  **准备数据集:**
    *   SOTA模型通常要求数据集遵循特定的格式（例如YOLOv8的YAML文件格式）。
    *   您需要为您的数据集创建一个符合其要求的`.yaml`配置文件，指明训练集、验证集、测试集的路径以及类别信息。这通常与我们项目中的`uav_dataset.yaml`类似。

3.  **训练SOTA模型:**
    *   使用SOTA模型官方提供的训练脚本，在您的数据集上从头开始训练。
    *   **关键参数对齐:** 尽量确保训练的超参数与我们模型具有可比性，例如：
        *   `imgsz`: 图像尺寸
        *   `epochs`: 训练轮次
        *   `batch`: 批次大小
        *   `optimizer`: 优化器类型

4.  **评估所有模型:**
    *   **在相同的测试集上**（正常天气和极端天气）分别评估我们训练好的“完整鲁棒模型”和SOTA模型。
    *   记录下每个模型在两个测试集上的**mAP@0.5**。
    *   记录下每个模型的**参数量（Parameters）**和**GFLOPs**（通常由其框架提供）。
    *   （可选）编写一个简单的脚本来测试每个模型在单张GPU上的**推理速度（FPS）**。

5.  **汇总结果:**
    *   将SOTA模型的各项指标，以及我们自己模型的指标，填入下面的新版“实验结果总览表”。

## 7. 实验结果总览表 (最终版)

为了方便您撰写论文，建议使用以下表格来汇总所有实验结果。

| 类别 (Category) | 实验名称 (Experiment) | 模型参数量 (M) | GFLOPs | mAP@0.5 (正常天气) | mAP@0.5 (极端天气) | FPS (可选用) | 备注 (Notes) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **我们的模型** | **Baseline** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 本项目的标准模型 |
| | **+ Weather Aug Only** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 消融: 仅数据增强 |
| | **+ BG Suppression Only** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 消融: 仅背景抑制 |
| | **+ Freq Enhancement Only** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 消融: 仅频域增强 |
| | **+ Condition Aware Only** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 消融: 仅环境感知模块 |
| | **Lightweight Robust Model** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 探索轻量化的可能性 |
| | **Robust Model (Full)** | **请填写** | **请填写** | **请填写** | **请填写** | **请填写** | **我们最终提出的模型** |
| **SOTA对比** | **YOLOv8n (Comparison)** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 使用 `run_comparison.sh` 运行 |
| | **YOLOv9-c (示例)** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | |
| | **YOLOv10-n/s/m (示例)** | *请填写* | *请填写* | *请填写* | *请填写* | *请填写* | 来自THU |

通过对比这张总览表，您可以从多个维度清晰地展示出您的`Robust Model`相较于基线和其他SOTA模型的优越性，这将构成您论文中最有力的证据。
