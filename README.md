# 面向大视场长距离深度感知的信息最优超表面成像

GitHub 仓库：[lsdfd/MetaDeepth](https://github.com/lsdfd/MetaDeepth)

这个仓库用于实现一个从物理光学仿真到深度重建的科研原型。当前第一版目标是先跑通：

```text
点源 PSF 计算 -> PSF Fisher 信息 -> RGB-D 编码图仿真 -> 神经网络恢复深度
```

第一阶段不直接追求完整真实硬件，而是先建立一个物理一致、可检查、可扩展的单波长窄带仿真系统。

## 目录结构

```text
.
├── docs/                 # 中文推导、研究计划、实验记录
├── references/           # 论文 PDF
├── translated_references/ # 翻译后的论文资料
├── configs/              # 实验参数配置
├── data/
│   ├── raw/              # 原始数据集
│   ├── processed/        # 清洗、裁剪、归一化后的数据
│   └── psf_cache/        # 预计算 PSF 表
├── src/
│   ├── optics/           # 光学传播、孔径、相位、PSF
│   ├── fisher/           # PSF 级和图像级 Fisher 信息
│   ├── simulation/       # RGB-D 到 CMOS 编码图的 forward model
│   ├── models/           # 深度恢复网络
│   └── utils/            # 配置、单位、可视化等工具
├── scripts/              # 可直接运行的实验入口
├── outputs/              # PSF 图、Fisher 图、编码图、训练结果
├── tests/                # 基础检查
├── environment.yml       # Conda 环境
├── requirements.txt      # Pip 依赖
└── pyproject.toml        # Python 项目配置
```

## 环境安装

之后所有代码都建议在 Conda 环境中运行，不使用系统 Python。

```bash
conda env create -f environment.yml
conda activate metasurface-depth
```

如果已经创建过环境，更新依赖：

```bash
conda env update -f environment.yml --prune
conda activate metasurface-depth
```

也可以在已激活的 Conda 环境中使用 pip 安装：

```bash
pip install -r requirements.txt
```

## 第一版核心依赖

光学计算和相位优化：

```text
torch          # 可微计算、FFT 传播、相位优化
torchoptics    # 可微 Fourier optics，后续用于 PSF 主实现或校验
waveprop       # 标量衍射传播校验
numpy/scipy    # 数值计算、插值、辅助验证
```

数据和图像：

```text
h5py
opencv-python
pillow
scikit-image
```

训练与可视化：

```text
matplotlib
tqdm
einops
torchvision
```

后续真实 meta-atom / RCWA 建库会单独加入 `torcwa` 或其他 RCWA 工具，不放在第一版必装依赖里。

## 可参考的开源项目

当前项目仍以本文档和 `docs/` 中的物理推导为准，下面项目只作为实现参考，不直接照搬实验路线：

```text
DFlat:
  https://github.com/DeanHazineh/DFlat
  可参考其 differentiable flat optics / metasurface imaging 的模块边界、传播和端到端优化写法。
  我们第一阶段只借鉴 PSF/传播组织方式，不先做端到端任务优化。

DepthFromDefocusWithLearnedOptics:
  https://github.com/computational-imaging/DepthFromDefocusWithLearnedOptics
  和“相位编码光学 + 单幅深度估计”背景接近，可参考其 PSF、coded aperture、RGB-D forward
  和深度网络连接方式。当前不进入训练阶段。
```

## 当前第一步

当前第一步是跑通一个点源 PSF sanity check：

```text
点源入射场
  -> 超表面孔径和相位调制
  -> angular spectrum 传播
  -> CMOS 面强度
  -> 归一化 PSF
```

运行：

```bash
conda activate metasurface-depth
python scripts/01_psf_sanity_check.py
```

期望输出：

```text
outputs/psf_figures/lens_phase_psf_grid.png
outputs/psf_figures/random_phase_psf_grid.png
```

第一步检查重点：

```text
1. PSF 是否非负；
2. PSF 归一化后总能量是否接近 1；
3. 不同深度下 PSF 是否变化；
4. 不同视场角下 PSF 是否变化。
```

## 第一阶段正式系统参数

见：

```text
configs/system_first_stage.yaml
```

当前参数不是 Nano-3D 对齐值，而是本项目自己的长距离、大视场深度感知目标：

```text
中心波长：590 nm
带宽：10 nm
FOV 扫描：30/60/90 deg
目标距离范围：1-10 m
目标距离采样：1/2/3/5/7/10 m
超表面口径扫描：3/5/10 mm
超表面-CMOS距离扫描：3/5/10/15 mm
CMOS：5472 x 3648，2.4 um pixel pitch
pupil grid main/debug：1024 / 512
PSF grid：128 x 128
```

`configs/baseline.yaml` 只保留为早期脚手架兼容文件，后续第一阶段实验默认以
`configs/system_first_stage.yaml` 为准。

## 数据集位置

第一阶段点源 PSF sanity check 不需要 RGB-D 数据集。后续做 RGB-D forward、
图像级验证和深度网络训练时，当前只采用两个数据源：

手动下载后放到：

```text
data/raw/hm3d/
data/raw/hypersim/
```

数据配置见：

```text
configs/datasets.yaml
docs/dataset_plan.md
```

查看第一阶段系统扫描表：

```bash
conda activate metasurface-depth
python scripts/00_print_system_sweep.py
```

当前计划使用：

```text
HM3D + Habitat：主数据源，1000 个真实 3D 场景，约 130 GB Habitat 场景包；
                 用 Habitat 渲染非连续 RGB-D 视角，可显式控制 FoV=30/60/90 deg。
Hypersim：补充数据源，461 个室内场景，公开 74619 张图，完整图像数据约 1.9 TB；
          提供 dense metric depth、相机参数和 scene split。
深度有效范围：优先筛 1-10 m
目标 FOV：30/60/90 deg
```

不再把 TartanAir / TartanGround / DIODE / NYU / KITTI 作为当前主数据路线；
它们只保留为历史调研背景，不进入本阶段数据方案。

本地先不拉完整数据集。第一批只准备约 100 张 Hypersim RGB-D pilot：

```text
data/raw/hypersim_pilot_100/
data/processed/hypersim_pilot_100/
```

用途：

```text
读取 RGB/depth；
检查 1-10 m depth mask；
做 RGB-D -> CMOS forward smoke test；
不作为训练集或泛化验证集。
```

旧 scaffold 参数曾包含：

```text
超表面口径默认：3 mm
口径扫描候选：3/5/10 mm
超表面-CMOS距离候选：37.6 mm
```

这些旧值不作为后续实验默认依据。第一阶段只验证透镜相位和随机相位的 PSF 行为，
不进入 Fisher 优化、RGB-D forward 或神经网络训练。
