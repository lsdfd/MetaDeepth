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
├── 参考文献/              # 论文 PDF
├── 翻译参考文献/          # 翻译后的论文资料
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

## 第一版 baseline 参数

见：

```text
configs/baseline.yaml
```

当前默认参数：

```text
中心波长：590 nm
带宽：10 nm
超表面口径：3 mm
口径扫描候选：3/5/10 mm
超表面-CMOS距离：37.6 mm
距离扫描候选：5/10/20/40 mm
主视场：30 deg
扩展视场扫描：30/60/90 deg
主深度范围：1-10 m
pupil grid：1024 x 1024
debug pupil grid：512 x 512
PSF grid：128 x 128
```

这些参数用于先建立单波长、点源级 PSF 仿真链路。第一阶段只验证透镜相位和随机相位的 PSF 行为，不进入 Fisher 优化、RGB-D forward 或神经网络训练。
