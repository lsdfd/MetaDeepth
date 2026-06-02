# MetaDeepth

面向大视场、长距离深度感知的信息最优超表面成像研究原型。

本仓库当前服务于一个从物理光学仿真到深度重建的科研流程：

```text
系统参数定义
-> 点源 PSF 仿真
-> PSF Fisher / nuisance-aware Fisher 分析
-> RGB-D 到编码 CMOS 图像仿真
-> 图像级 Fisher / 任务可分性验证
-> 神经网络深度重建
```

当前仍处在第一阶段。不要把现有 Fisher、RGB-D forward、深度网络 smoke test 当成最终实验结果；它们主要用于确认代码链路和梯度/数据接口。

GitHub: [lsdfd/MetaDeepth](https://github.com/lsdfd/MetaDeepth)

## 当前阶段

第一阶段目标是建立本项目自己的长距离、大视场系统参数和可信的单波长点源 PSF 仿真，不对齐 Nano-3D 的近距离设置。

第一阶段系统配置以 [configs/system_first_stage.yaml](configs/system_first_stage.yaml) 为准：

```text
中心波长：590 nm
带宽：10 nm
FoV：30 / 60 / 90 deg
目标距离范围：1-10 m
深度采样：1 / 2 / 3 / 5 / 7 / 10 m
超表面口径扫描：3 / 5 / 10 mm
超表面-CMOS 距离扫描：3 / 5 / 10 / 15 mm
CMOS：5472 x 3648，2.4 um pixel pitch
pupil grid：main 1024，debug 512
PSF grid：128
相位基线：lens / random / spiral / double_helix / multiring_spiral
```

`configs/baseline.yaml` 只是早期脚手架兼容文件，后续实验默认不要用它。

## 目录结构

```text
configs/                实验和数据配置
data/raw/               原始数据和小型 pilot 数据
data/processed/         后续处理后的数据
data/psf_cache/         预计算 PSF 缓存
docs/                   研究计划、推导和数据路线文档
references/             论文 PDF
translated_references/  翻译后的论文资料
outputs/                smoke test 输出、图和 CSV
scripts/                可运行实验入口
src/optics/             pupil、相位、传播、点源 PSF
src/fisher/             PSF Fisher 公式和 phase-control 工具
src/simulation/         RGB-D 到编码 CMOS forward model
src/models/             深度网络 scaffold、数据、loss、metrics
src/utils/              配置、单位、系统参数解析
```

## 环境

所有项目代码运行优先使用 conda 环境，不使用系统 Python：

```bash
conda env create -f environment.yml
conda activate metasurface-depth
```

如果环境已存在：

```bash
conda env update -f environment.yml --prune
conda activate metasurface-depth
```

主要依赖包括：

```text
torch / torchvision
torchoptics / waveprop
numpy / scipy / matplotlib
pyyaml / h5py / tqdm / einops
opencv-python / pillow / scikit-image
```

`torcwa` / RCWA / meta-atom library 暂不属于第一阶段依赖，后续进入 fabrication-aware 设计时再加入。

## 数据路线

当前数据路线只保留两个数据源：

```text
HM3D + Habitat
Hypersim
```

不要把 TartanAir、TartanGround、DIODE、NYU、KITTI、ScanNet 作为当前主数据路线。它们只保留为历史调研背景。

数据配置见 [configs/datasets.yaml](configs/datasets.yaml)，详细说明见 [docs/dataset_plan.md](docs/dataset_plan.md)。

```text
HM3D + Habitat：
  主数据源。
  1000 个真实 3D 场景，约 130 GB Habitat 场景包。
  可用 Habitat 渲染非连续 RGB-D 视角，并显式设置 FoV=30/60/90 deg。

Hypersim：
  补充数据源。
  461 个室内场景，公开 74619 张图，完整图像数据约 1.9 TB。
  提供 dense metric depth、相机参数和 scene split。
```

本地当前不拉完整大数据集。仓库已包含一个小型 Hypersim pilot：

```text
data/raw/hypersim_pilot_100/
```

pilot 当前形态：

```text
10 个 scene
100 张 RGB-D
RGB: frame.*.tonemap.jpg, 1024 x 768
Depth: frame.*.depth_meters.hdf5, metric depth
相机参数: metadata_camera_parameters.csv
FoV: horizontal 约 60 deg，vertical 约 46.8 deg
```

这个 pilot 只用于数据读取、depth mask、可视化和 RGB-D forward smoke test，不用于训练或泛化评价。90 deg FoV 数据后续应通过 HM3D/Habitat 渲染获得。

重新下载 pilot：

```bash
conda activate metasurface-depth
python scripts/02_download_hypersim_pilot_100.py \
  --out data/raw/hypersim_pilot_100 \
  --scenes 10 \
  --frames-per-scene 10
```

## 运行入口

查看第一阶段系统扫描表：

```bash
conda activate metasurface-depth
python scripts/00_print_system_sweep.py
```

点源 PSF sanity check：

```bash
python scripts/01_psf_sanity_check.py
```

当前 PSF sanity 使用 `configs/system_first_stage.yaml`，会跑 lens/random 等相位基线的 debug case，并输出数值摘要到：

```text
outputs/psf_figures/first_stage/psf_sanity_debug_summary.csv
```

PSF Fisher baseline：

```bash
python scripts/02_psf_fisher_experiment.py --mode baseline
```

输出：

```text
outputs/fisher/first_stage/baseline_fisher_summary.csv
outputs/fisher/first_stage/baseline_psf_similarity_depth.csv
```

小型 phase-control 优化 smoke test：

```bash
python scripts/02_psf_fisher_experiment.py --mode optimize
```

输出：

```text
outputs/fisher/first_stage/optimized_phase.pt
outputs/fisher/first_stage/optimized_phase_history.csv
outputs/fisher/first_stage/optimized_phase.png
```

RGB-D 到编码 CMOS 图像 forward / 图像级 Fisher smoke test：

```bash
python scripts/03_image_forward_fisher_experiment.py \
  --data-root data/raw/hypersim_pilot_100 \
  --sample-index 0 \
  --patch-size 8 \
  --psf-size 31 \
  --grid 64 \
  --phase random
```

输出：

```text
outputs/image_forward/first_stage/image_forward_fisher_summary.csv
outputs/image_forward/first_stage/image_forward_smoke.png
```

深度学习 scaffold smoke train/eval：

```bash
python scripts/04_depth_learning.py --mode train
python scripts/04_depth_learning.py --mode eval
```

配置见 [configs/depth_learning.yaml](configs/depth_learning.yaml)。当前训练数据是 synthetic smoke dataset，不是真实实验数据。

输出：

```text
outputs/depth_learning/smoke/depth_model.pt
outputs/depth_learning/smoke/train_metrics.csv
outputs/depth_learning/smoke/eval_metrics.csv
```

## 代码模块

光学模块：

```text
src/optics/pupil.py
  make_xy_grid、circular_aperture、exact_lens_phase、random/spiral/double_helix/multiring_spiral phase templates

src/optics/propagation.py
  angular spectrum 标量传播

src/optics/psf.py
  spherical point-source field、point-source PSF 计算、中心裁剪和归一化
```

Fisher 模块：

```text
src/fisher/psf_fisher.py
  autograd Jacobian
  Poisson Fisher
  nuisance-aware effective depth Fisher
  CRLB
  low-dimensional phase control
  PSF similarity metrics
```

RGB-D forward 模块：

```text
src/simulation/rgbd_forward.py
  Hypersim pilot sample loader
  green-channel intensity
  ray angle map
  direct spatially varying PSF rendering
  scalar patch-depth Fisher
  Poisson Mahalanobis metric
```

深度模型模块：

```text
src/models/simple_unet.py
  compact DepthUNet / FastDepth-style baseline
  FoundationDepthAdapter placeholder

src/models/depth_data.py
  SyntheticDepthDataset
  ManifestDepthDataset hook

src/models/depth_losses.py
  masked L1 / SiLog combination

src/models/depth_metrics.py
  AbsRel / RMSE / MAE / delta1
```

## 当前注意事项

1. 当前 PSF 数值仍处在 debug 阶段。`grid=512, aperture=3 mm, d=10 mm, padding=2` 时 lens phase 相邻采样相位步长可能过大，lens PSF 有可见采样伪影。扩展 Fisher 或解读结果前，必须复查采样、传播距离、crop 策略。
2. spiral / double_helix / multiring_spiral 只是传统相位模板 baseline，不是某篇论文的最终优化 mask。
3. `src/simulation/rgbd_forward.py` 的 spatially varying PSF rendering 是公式直写版本，故意慢，适合小 patch 验证；第一阶段不要先替换成 depth-bin convolution。
4. `outputs/` 中的 Fisher、image-forward、depth-learning 文件都是 smoke test 产物，不是最终实验图表。
5. 后续真正训练深度网络前，需要先把 HM3D/Habitat 或 Hypersim 数据整理成项目统一 manifest：

```text
rgb: H x W x 3, float32, [0, 1]
depth_m: H x W, float32, meters
valid_mask: H x W, bool, depth in [1, 10] m
intrinsics or ray-angle model
horizontal/vertical FoV
scene_id / frame_id
```

## 参考文档

实现和实验顺序优先参考：

```text
docs/面向大视场长距离深度感知的信息最优超表面成像研究计划.tex
docs/超表面深度成像项目公式推导整理.tex
docs/metasurface_depth_imaging_questions.tex
docs/information_optimal_metasurface_depth.tex
docs/dataset_plan.md
```

当前仓库是研究原型，不是稳定软件包。代码结构会随着物理模型、采样策略和实验设计继续调整。
