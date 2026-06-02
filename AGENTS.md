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
baseline parameter table.
```

So the immediate task is not Fisher optimization, not full RGB-D rendering, and not neural network training. The first coding milestone is:

```text
Fixed first-version system config + reliable single-wavelength point-source PSF simulator
```

## First Baseline Parameters

The first-version system should prioritize the user's target task, not simply copy Nano-3D near-field settings. Nano-3D-like numbers may be used only as hardware references, not as the project target.

```text
center wavelength: 590 nm
bandwidth: 10 nm
target FOVs: 30 deg, 60 deg, 90 deg
target distance range: 1-10 m
aperture diameter: initially keep 3 mm as one candidate, but include aperture scans such as 3/5/10 mm if feasible
metasurface-CMOS distance: initially keep 37.6 mm as one candidate, but include distance scans such as 5/10/20/40 mm if feasible
sensor reference: 5472 x 3648, 2.4 um pixel pitch
system-level pupil grid: 1024 x 1024 for baseline
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

The environment has not necessarily been created yet. Check before running scripts:

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
参考文献/              source papers
翻译参考文献/          translated papers
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

```text
PSF is nonnegative;
normalized PSF sum is close to 1;
depth changes modify the PSF;
field angle changes modify the PSF;
ordinary lens baseline behaves plausibly;
random phase baseline produces a coded/speckle-like response;
sampling changes do not create obvious numerical artifacts.
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
src/optics/pupil.py
src/optics/propagation.py
src/optics/psf.py
src/fisher/psf_fisher.py
src/simulation/rgbd_forward.py
src/models/simple_unet.py
src/utils/config.py
src/utils/units.py
scripts/01_compute_psf_demo.py
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
1. Make configs/baseline.yaml strictly match the research plan baseline.
2. Create/activate the conda environment.
3. Implement a clearer first-stage PSF sanity script:
   scripts/01_psf_sanity_check.py
4. Generate lens-phase and random-phase PSFs for:
   depths = [0.2, 0.4, 0.8, 1.2] m
   angles = [-15, 0, 15] deg
   lambda = 590 nm
5. Save figures to outputs/psf_figures/.
```

Do not proceed to Fisher until this first-stage PSF behavior has been reviewed.
