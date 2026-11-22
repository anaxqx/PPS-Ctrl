# PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation

This repository contains the ControlNet training script and per-pixel shading modules for [PPS-Ctrl](https://arxiv.org/abs/2504.17067). The code release is in progress.

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

### 2. Prepare Data

Download the following datasets:

[SimCol3D](https://www.ucl.ac.uk/interventional-surgical-sciences/simcol3d-3d-reconstruction-during-colonoscopy-challenge)

[C3VD](https://durrlab.github.io/C3VD/)

[Colon10K](https://endoscopography.web.unc.edu/place-recognition-in-colonoscopy/)

Set `input_dir` and `output_dir` in `utils/compute_pps.py`, then run `python utils/compute_pps.py` to compute PPS maps.

### 3. Train

For `train_controlnet.py`, set `--pretrained_model_name_or_path` and pass your dataset with `--dataset_name` or `--train_data_dir`. Use `--image_column`, `--conditioning_image_column`, and `--caption_column` to match the columns for target images, PPS maps, and captions.

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
