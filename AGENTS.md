# AGENTS.md

This file is the handoff guide for Codex/agents working on this repository.

## Project Context

The project is a research prototype for information-optimal metasurface depth imaging. The intended pipeline is:

```text
system parameter definition
-> point-source PSF simulation
-> PSF-level Fisher / nuisance-aware Fisher optimization
-> RGB-D encoded CMOS image simulation
-> image-level Fisher / task validation
-> neural depth reconstruction
```

The project is currently at the very beginning of the coding phase. Existing code is a scaffold and is not final. It may be reorganized when the research plan and physical model require it.

## Authoritative Documents

Before making major design or implementation decisions, read these files:

```text
docs/面向大视场长距离深度感知的信息最优超表面成像研究计划.tex
docs/超表面深度成像项目公式推导整理.tex
docs/metasurface_depth_imaging_questions.tex
docs/information_optimal_metasurface_depth.tex
docs/dataset_plan.md
```

The most important one for implementation order is:

```text
docs/面向大视场长距离深度感知的信息最优超表面成像研究计划.tex
```

Do not invent a new research order. Follow the stage plan in that document.

## Current Stage

The current stage is the first stage from the research plan, with the user's corrected target system rather than a Nano-3D near-range target:

```text
Weeks 1-2:
Fix first-version system parameters:
target FOVs = 30 deg, 60 deg, 90 deg
target distance range = 1-10 m
center wavelength initially = 590 nm unless revised
metasurface aperture / period / sensor distance / CMOS mapping must be explicitly specified and scanned where needed

Build single-wavelength PSF simulation.

Expected outputs:
point-source PSF computation code;
ordinary lens / random phase PSFs;
system parameter table.
```

So the immediate task is not Fisher optimization, not full RGB-D rendering, and not neural network training. The first coding milestone is:

```text
Fixed first-version system config + reliable single-wavelength point-source PSF simulator
```

Important user correction on 2026-06-02: do not jump to PSF-level Fisher, RGB-D datasets, or neural reconstruction before the single-point and discrete field/depth PSF simulator is physically credible. Future planning must start from the physics in the docs, especially the point-source PSF equations, not from available code scaffolds.

## First-Stage System Parameters

The first-version system must prioritize the user's target task, not Nano-3D. Nano-3D-like numbers are only literature references and must not drive the default configuration.

```text
center wavelength: 590 nm
bandwidth: 10 nm
target FOVs: 30 deg, 60 deg, 90 deg
target distance range: 1-10 m
depth samples: 1/2/3/5/7/10 m
aperture scan: 3/5/10 mm
metasurface-CMOS distance scan: 3/5/10/15 mm
sensor reference: 5472 x 3648, 2.4 um pixel pitch
system-level pupil grid: 1024 x 1024 for main runs
fast debug pupil grid: 512 x 512 is acceptable for quick tests
PSF output grid: initially 128 x 128 or cropped from propagation grid
```

Important: the pupil grid is a system-level effective phase sampling grid. It is not the true number of meta-atoms. A 3 mm aperture with 400 nm period has about 7500 x 7500 meta-atoms.

## Environment

Use conda for all future code execution. Do not use the system Python for project runs.

Environment files:

```text
environment.yml
requirements.txt
pyproject.toml
```

Recommended setup:

```bash
conda env create -f environment.yml
conda activate metasurface-depth
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate metasurface-depth
```

Environment status as of 2026-06-02: `metasurface-depth` has been created at `/opt/homebrew/Caskroom/miniforge/base/envs/metasurface-depth`. Import smoke check passed for `torch`, `torchoptics`, `waveprop`, `cv2`, and `numpy`.

Check before running scripts:

```bash
conda env list
```

Current first-version dependencies include:

```text
torch
torchvision
torchoptics
waveprop
numpy
scipy
matplotlib
pyyaml
h5py
tqdm
einops
opencv-python
pillow
scikit-image
```

`torcwa` / RCWA tools are not first-stage dependencies. Add them later only when moving from effective phase design to meta-atom library / fabrication-aware design.

GitHub remote:

```text
https://github.com/lsdfd/MetaDeepth
```

## Repository Layout

```text
docs/                 Chinese derivations, research plans, notes
references/           source papers
translated_references/ translated papers
configs/              YAML experiment configs
data/raw/             raw datasets
data/processed/       cleaned RGB-D data
data/psf_cache/       precomputed PSF tables
src/optics/           optical grids, sources, propagation, PSF
src/fisher/           PSF-level and image-level Fisher code
src/simulation/       RGB-D to encoded CMOS forward model
src/models/           neural depth reconstruction models
src/utils/            config, units, visualization, metrics
scripts/              runnable experiment scripts
outputs/              generated figures, maps, encoded images, checkpoints
tests/                minimal sanity checks
```

## Coding Rules for This Project

1. Follow the research plan order. Do not jump to Fisher, RGB-D datasets, or networks before the baseline PSF simulation is credible.
2. Treat existing code as a scaffold, not as fixed architecture.
3. Keep parameters in YAML configs instead of hard-coding them in scripts.
4. Prefer PyTorch for optical propagation and phase optimization because later gradients with respect to phase are needed.
5. Use `torchoptics` and `waveprop` as useful optical libraries/checks, but do not make the physics opaque.
6. Fisher code should be implemented in this project with PyTorch. There is no expected off-the-shelf Fisher package for the project's custom parameterization.
7. The first PSF simulator should compare at least ordinary lens phase and random phase.
8. Always verify PSF sanity:

Additional user preference from 2026-06-02: keep the code simple and direct. Do not create extra files, wrappers, or abstraction layers unless they clearly reduce complexity or are necessary for correctness. Prefer concise, readable implementations that follow the physics formulas exactly; avoid "vibe coding" structure that scatters one idea across many files.

```text
PSF is nonnegative;
normalized PSF sum is close to 1;
depth changes modify the PSF;
field angle changes modify the PSF;
ordinary lens baseline behaves plausibly;
random phase baseline produces a coded/speckle-like response;
sampling changes do not create obvious numerical artifacts.
```

For the first-stage PSF work, explicitly reason about discretizing the target space before writing Fisher code:

```text
depth range: 1-10 m, sampled initially at 1/2/3/5/7/10 m;
FOVs: 30/60/90 deg;
angle samples per FOV: directly use 2D center/cross/corner samples, not a 1D-only horizontal sweep;
phase baselines: ordinary lens and random phase first;
aperture and metasurface-CMOS distance scans are part of PSF feasibility, not Fisher optimization yet.
```

## Conversation Handoff Rule

After each meaningful Codex/agent conversation or implementation step, update this `AGENTS.md` file if any of the following changed:

```text
current research stage;
agreed implementation order;
baseline parameters;
environment setup status;
important user preferences;
new files or directories;
known blockers;
next immediate step.
```

The update should be concise and practical. This file is not a lab notebook; it is a live handoff guide so the next Codex/agent knows what to do without rereading the whole conversation.

When updating this file, do not rewrite the whole document unless the project direction changed substantially. Prefer small edits to the relevant sections.

## Current Files Already Created

Scaffold files currently include:

```text
.gitignore
configs/baseline.yaml
configs/system_first_stage.yaml
configs/datasets.yaml
src/optics/pupil.py
src/optics/propagation.py
src/optics/psf.py
src/fisher/psf_fisher.py
src/simulation/rgbd_forward.py
src/models/simple_unet.py
src/utils/config.py
src/utils/system_config.py
src/utils/units.py
scripts/01_compute_psf_demo.py
scripts/00_print_system_sweep.py
scripts/01_psf_sanity_check.py
README.md
environment.yml
requirements.txt
pyproject.toml
```

These files are not final. The next agent may modify or reorganize them to better match the research plan.

## Immediate Next Step

The next coding step should be:

```text
1. Build and verify a physically clear single-point PSF path:
   point source -> spherical wave at metasurface -> aperture + phase -> propagation -> CMOS intensity.
2. Make scripts/01_psf_sanity_check.py read configs/system_first_stage.yaml.
3. First run one reviewable point:
   phase = lens/random
   FOV = 90 deg, angle = center/cross/corner cases
   depth = one near point and one far point within 1-10 m
   aperture = 3 mm
   metasurface-CMOS distance = one configured candidate
4. Then expand to discrete PSF sweeps:
   FOVs = 30/60/90 deg
   depths = 1/2/3/5/7/10 m
   aperture candidates = 3/5/10 mm
   metasurface-CMOS distance candidates = 3/5/10/15 mm
5. Save figures and numerical sanity summaries to outputs/psf_figures/first_stage/.
```

Do not proceed to Fisher until this first-stage PSF behavior has been reviewed.

Current PSF implementation status as of 2026-06-02:

```text
scripts/01_psf_sanity_check.py now runs a debug point-source PSF sanity check from configs/system_first_stage.yaml.
Debug case currently uses FOV=90 deg, 2D center/cross/corner angles, depths 1 and 10 m, aperture 3 mm, metasurface-CMOS distance 10 mm, and lens/random phase baselines.
The lens baseline uses exact equal-optical-path focusing phase, not the paraxial thin-lens phase.
The point-source field follows the documented spherical-wave formula R_o and exp(i k R_o)/R_o.
Known numerical issue: with grid=512, aperture=3 mm, d=10 mm, and padding=2, the lens phase neighbor step is about 18 rad, so the lens PSF has visible sampling artifacts. Before trusting lens PSFs or expanding to Fisher, adjust sampling/grid/distance/crop policy and review the generated PSF figures and CSV summary.
```

Current Fisher implementation status as of 2026-06-02:

```text
src/fisher/psf_fisher.py contains the minimal PSF Fisher formulas: autograd Jacobian for theta=[z, alpha_x, alpha_y, photon_count], Poisson Fisher, nuisance-aware effective depth Fisher, CRLB, low-dimensional phase control, and PSF similarity metrics.
scripts/02_psf_fisher_experiment.py is the single Fisher entry point with --mode baseline / optimize / report.
Baseline smoke run completed for FOV=90 deg, 2D center/cross/corner angles, depths 1/3/5/7/10 m, aperture 3 mm, d=10 mm, lens/random phases. Outputs: outputs/fisher/first_stage/baseline_fisher_summary.csv and baseline_psf_similarity_depth.csv.
Optimization smoke run completed for a small low-dimensional phase control grid and saved outputs/fisher/first_stage/optimized_phase.pt, optimized_phase.png, and optimized_phase_history.csv. This is only a gradient-chain smoke test, not a final optimized metasurface design.
Important preliminary result: center-view PSFs for 5/7/10 m are nearly identical under the current debug parameters, while some corner-view PSFs differ more strongly. Do not over-interpret until sampling/crop/distance issues are reviewed.
```

Current baseline/image-forward status as of 2026-06-02:

```text
Traditional phase baselines now include lens, random, spiral, double_helix, and multiring_spiral. They are simple template baselines, not exact reproduction of a specific paper's optimized mask.
scripts/02_psf_fisher_experiment.py baseline smoke completed with 5 phase types and produced 225 Fisher rows.
src/simulation/rgbd_forward.py now contains direct per-pixel spatially varying PSF rendering for small RGB-D patches, following Y(u,v)=sum_x sum_y I(x,y) h(u,v; x,y, D, alpha_x, alpha_y). This is intentionally slow but formula-direct; do not replace it with depth-binned convolution for the first image-level validation.
scripts/03_image_forward_fisher_experiment.py loads the local Hypersim pilot data, renders a small patch, computes depth perturbation distances, scalar patch-depth Fisher, SSIM, frequency/L1/JS/Mahalanobis metrics, and a tiny probe smoke test.
Image-forward smoke ran with patch_size=6, psf_size=15, grid=32, phase=random. The chosen center patch was nearly dark, so results are only a pipeline smoke test. Next improve patch selection before interpreting image-level numbers.
```

Current depth-learning scaffold status as of 2026-06-02:

```text
The depth reconstruction code now has a deliberately minimal scaffold for the later stage, without treating it as the current physics bottleneck.
configs/depth_learning.yaml defines a smoke setup with input_mode = rgb / meta / rgb_meta support.
src/models/simple_unet.py contains DepthUNet, a compact UNet/FastDepth-style skip-fusion metric-depth baseline constrained to the 1-10 m range, plus a placeholder FoundationDepthAdapter for later Depth Anything / DPT / MiDaS style backbones.
src/models/depth_data.py contains a SyntheticDepthDataset for train/eval smoke tests and a not-yet-implemented ManifestDepthDataset hook for future processed HM3D/Hypersim/metasurface encoded data.
src/models/depth_losses.py and src/models/depth_metrics.py contain masked L1, SiLog, AbsRel, RMSE, MAE, and delta1 metrics.
scripts/04_depth_learning.py is the single train/eval entry point with --mode train / eval, and saves outputs/depth_learning/smoke/depth_model.pt plus CSV metrics.
Smoke checks passed in conda env metasurface-depth on 2026-06-02.
Do not interpret the synthetic smoke metrics as experimental results. The next real step for learning is to define a processed RGB-D/Meta manifest after the PSF sampling and RGB-D forward model are physically credible.
```

Dataset note: the active dataset plan is limited to HM3D/Habitat and Hypersim. Do not reintroduce TartanAir, TartanGround, DIODE, NYU, KITTI, or ScanNet as primary datasets unless the user explicitly changes direction.

Manual downloads should be placed under:

```text
data/raw/hm3d/
data/raw/hypersim/
```

Current dataset scale notes:

```text
HM3D/Habitat: 1000 3D scenes, about 130 GB Habitat scene archive; render RGB-D views with explicit FoV=30/60/90 deg.
Hypersim: 461 indoor scenes, 74619 public images, about 1.9 TB full image dataset; dense metric depth and camera metadata.
```

Do not download full datasets for initial tests. First local dataset target is a Hypersim pilot subset:

```text
about 100 RGB-D images
raw path: data/raw/hypersim_pilot_100/
processed path: data/processed/hypersim_pilot_100/
purpose: reader/mask/visualization/RGB-D forward smoke tests only
budget: keep under about 3 GB
```
