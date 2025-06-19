# PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation

🚧 **This repository is currently under construction, and we are continually developing and refining the method. Check back soon for updates!** 🚧

🚧 **We will provide updated complete version of code and checkpoints upon paper acceptance** 🚧

This repository contains the code and models for our paper:

> **PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation**  

## Overview

**PPS-Ctrl** is a image translation framework that combines **Stable Diffusion** and **ControlNet**, guided by a **Per-Pixel Shading (PPS)** map — a physics-informed representation capturing surface-light interactions. Unlike prior sim-to-real approaches that condition on depth maps, PPS provides a more faithful and geometrically consistent structural prior, enabling better texture realism and structure preservation in endoscopy image translation.


---

## Getting Started

### 1. Environment Setup

We recommend Python 3.9 with PyTorch ≥ 2.0 and the HuggingFace `diffusers` library.

```bash
conda create -n ppsctrl python=3.9
conda activate ppsctrl
pip install -r requirements.txt
```
### 2. Prepare Data

Download the following datasets:

[SimCol3D](https://www.ucl.ac.uk/interventional-surgical-sciences/simcol3d-3d-reconstruction-during-colonoscopy-challenge)

[C3VD](https://durrlab.github.io/C3VD/)

[Colon10K](https://endoscopography.web.unc.edu/place-recognition-in-colonoscopy/)

Precompute PPS maps:

```bash
python utils/compute_pps.py --depth_dir path/to/depth --output_dir path/to/pps
```

### 3. Train

#### Stage 1: Fine-tune Stable Diffusion

```bash
bash scripts/train_sd.sh
```

#### Stage 2: Train ControlNet with PPS conditioning

```bash
bash scripts/train_controlnet.sh
```

### 4. Inference

```bash
python scripts/infer.py --depth path/to/test/depth --output path/to/save
```

## Pretrained Models
We provide pretrained checkpoints for:

Stable Diffusion (domain-finetuned)

ControlNet with PPS-encoder-decoder

📥 Download links coming soon!

## Citation


## Acknowledgments

This work builds on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and [PPSNet](https://github.com/yahskapar/PPSNet/tree/main). We thank the maintainers for their open-source contributions.
