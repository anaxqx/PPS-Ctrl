# PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation

**Code release in progress:** the repository includes a ControlNet training script and PPS modules; prepared data and project-specific run configuration are not included.

This repository contains code for our paper:

> **PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation**

## Overview

**PPS-Ctrl** is an image translation framework that combines **Stable Diffusion** and **ControlNet**, guided by a **Per-Pixel Shading (PPS)** map — a physics-informed representation capturing surface-light interactions. Unlike prior sim-to-real approaches that condition on depth maps, PPS provides a more faithful and geometrically consistent structural prior, enabling better texture realism and structure preservation in endoscopy image translation.

---

## Getting Started

### 1. Environment Setup

We recommend Python 3.9 with PyTorch ≥ 2.0 and the HuggingFace `diffusers` library.

```bash
conda create -n ppsctrl python=3.9
conda activate ppsctrl
```

The repository does not currently include a populated dependency file; install the required packages for your environment.

### 2. Prepare Data

Download the following datasets:

[SimCol3D](https://www.ucl.ac.uk/interventional-surgical-sciences/simcol3d-3d-reconstruction-during-colonoscopy-challenge)

[C3VD](https://durrlab.github.io/C3VD/)

[Colon10K](https://endoscopography.web.unc.edu/place-recognition-in-colonoscopy/)

Precompute PPS maps with `utils/compute_pps.py`. The script currently uses hard-coded input and output paths, which must be set in the script before use.

### 3. Train

The training entry point is `train_controlnet.py`. Configure the model and dataset arguments for your run; a project-specific launch command is not currently provided.

## Citation

```bibtex
@article{xiong2025pps,
  title={PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation},
  author={Xiong, Xinqi and Beltran, Andrea Dunn and Choi, Jun Myeong and Niethammer, Marc and Sengupta, Roni},
  journal={arXiv preprint arXiv:2504.17067},
  year={2025}
}
```

## Acknowledgments

This work builds on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), [Hugging Face Diffusers](https://github.com/huggingface/diffusers) and [PPSNet](https://github.com/yahskapar/PPSNet/tree/main). We thank the maintainers for their open-source contributions.
