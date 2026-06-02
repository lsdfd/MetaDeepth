# 第一阶段后续 RGB-D 数据路线

当前数据路线只保留两个数据源：

```text
HM3D + Habitat
Hypersim
```

TartanAir、TartanGround、DIODE、NYU Depth V2、KITTI、ScanNet 暂不作为当前主数据集。

## 选择原则

后续要训练深度学习模型，因此数据集不能只是一段连续视频轨迹。当前优先级是：

```text
1. 多场景，而不是单轨迹连续帧；
2. 按 scene 做 train/val/test split；
3. 有 metric depth；
4. 能覆盖或生成 FoV = 30/60/90 deg；
5. 有成熟生态，后续可以稳定读数据、渲染、训练和复现实验。
```

## HM3D + Habitat

定位：

```text
主数据源
```

关键信息：

```text
场景数：1000 个真实 3D 场景
公开 split：train 800 / val 100 / test held-out 100
下载体量：约 130 GB Habitat 场景包
数据类型：带纹理的真实 3D scene scans
生态：Habitat-Sim / Habitat-Lab / HM3D / HM3D-Semantics
```

为什么适合本项目：

```text
1. 可以从 3D 场景中随机采样非连续视角；
2. 可以显式设置相机 FoV = 30/60/90 deg；
3. 可以渲染 RGB + metric depth；
4. 可以按 scene split 验证泛化；
5. 深度可以统一 mask 到 1-10 m。
```

本地路径：

```text
data/raw/hm3d/
data/processed/hm3d/
data/processed/hm3d_rendered_rgbd/
```

建议先下载：

```text
HM3D minival / val 小规模 split
```

先验证 Habitat 安装、RGB-D 渲染、FoV 控制和 1-10 m depth mask，再决定是否拉完整 train。

## Hypersim

定位：

```text
补充数据源，主要用于 dense indoor RGB-D 训练和验证
```

关键信息：

```text
场景数：461 个室内场景
原始渲染图像数：77400
公开图像数：74619
完整图像数据体量：约 1.9 TB
数据类型：photorealistic synthetic indoor RGB-D
深度定义：distance from camera center in meters
许可：CC-BY-SA-3.0
```

为什么适合本项目：

```text
1. dense metric depth；
2. 有完整相机参数和位姿；
3. 有官方下载脚本和 metadata；
4. 有 scene-level split；
5. 可作为 HM3D 的室内高质量补充。
```

本地路径：

```text
data/raw/hypersim/
data/processed/hypersim/
```

## 本地 Pilot 子集

完整 HM3D/Hypersim 都很大，当前不直接全量下载。先准备一个约 100 张图的
Hypersim pilot：

```text
data/raw/hypersim_pilot_100/
data/processed/hypersim_pilot_100/
```

目标：

```text
图像数：约 100 张 RGB-D
建议场景：10 个 scene，每个 scene 10 张
预算：优先控制在 3 GB 内
来源：Hypersim 官方数据，优先用按 ZIP 内文件子集下载的方式
```

这个 pilot 只用于：

```text
1. 测试 RGB/depth 读取；
2. 测试 depth_meters 和 1-10 m mask；
3. 测试相机参数读取；
4. 测试 RGB-D -> CMOS forward smoke test；
5. 快速画图检查数据是否合理。
```

这个 pilot 不用于：

```text
1. 网络训练；
2. 泛化验证；
3. 写论文结果。
```

Pilot 当前数据形态：

```text
RGB：frame.*.tonemap.jpg，1024 x 768
Depth：frame.*.depth_meters.hdf5，metric depth
相机参数：metadata_camera_parameters.csv
settings_camera_fov：1.04719758 rad，约 60 deg
估算视场：horizontal FoV 约 60 deg，vertical FoV 约 46.8 deg
```

适配性：

```text
适合：
  1. RGB/depth 文件读取；
  2. depth 1-10 m mask；
  3. 30 deg / 60 deg FoV smoke test；
  4. RGB-D -> encoded CMOS forward pipeline 调试。

不适合：
  1. 90 deg FoV 验证；
  2. 最终训练；
  3. 泛化评价。
```

90 deg FoV 数据应由 HM3D/Habitat 渲染获得，因为 Habitat 可以显式设置目标 FoV。

下一步需要做一个标准化 loader，把 raw pilot 转成项目内部统一格式：

```text
rgb: H x W x 3, float32, [0, 1]
depth_m: H x W, float32, meters
valid_mask: H x W, bool, depth in [1, 10] m
intrinsics: fx/fy/cx/cy or camera ray model
fov_deg: horizontal/vertical FoV
scene_id / frame_id
```

## 当前不选其他数据集的原因

```text
TartanAir / TartanGround：
  有 90 deg 和机器人视角，但主要是轨迹数据；可用但容易回到连续帧问题。

DIODE：
  真实 RGB-D 很好，但 FoV 约 60 x 45 deg，不适合 90 deg 主验证；规模和生态不够做主训练源。

NYU Depth V2 / KITTI：
  benchmark 生态强，但不是本项目主目标数据形态；可作为论文附加 benchmark，不作为当前主路线。

ScanNet：
  生态强，但仍是 RGB-D video scan 形态；当前先用 HM3D/Habitat 控制非连续视角和 FoV。
```

## 后续采样策略

```text
训练/验证按 scene split，不按帧随机切分；
HM3D 从每个场景随机采样非连续视角；
每个视角分别渲染 FoV = 30/60/90 deg；
所有 depth 统一保留 1-10 m 有效范围；
先小规模渲染验证 forward pipeline，再扩大训练集。
```
