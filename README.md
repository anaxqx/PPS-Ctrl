# PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation

This repository accompanies the paper **PPS-Ctrl: Controllable Sim-to-Real Translation for Colonoscopy Depth Estimation** (arXiv:2504.17067). It contains an in-progress code release for a ControlNet-based image-translation method conditioned on per-pixel shading (PPS).

> **Repository status:** This is not currently a complete, ready-to-run reproduction package. The repository does not include a populated dependency manifest, data preparation guide, pretrained checkpoints, or a verified end-to-end command. The paper describes the method; the code here is a partial implementation.

## What is in the repository

- `train_controlnet.py`: a ControlNet training script adapted from the Diffusers example. It expects model and dataset configuration through command-line arguments; the repository does not provide a project-specific launch command or prepared dataset.
- `pps-ctrl/pps_ctrl.py` and `pps-ctrl/pps_ctrl_pipeline.py`: PPS ControlNet and pipeline code.
- `utils/compute_pps.py`: utilities for deriving PPS values from depth maps. Its current script entry point uses hard-coded local paths and does not accept the `--depth_dir` or `--output_dir` options shown in older instructions.

## Data and setup

The paper uses the [SimCol3D](https://www.ucl.ac.uk/interventional-surgical-sciences/simcol3d-3d-reconstruction-during-colonoscopy-challenge), [C3VD](https://durrlab.github.io/C3VD/), and [Colon10K](https://endoscopography.web.unc.edu/place-recognition-in-colonoscopy/) datasets. This repository does not bundle those datasets or provide a complete staging/preprocessing workflow.

No working environment setup command is provided yet. In particular, the repository currently has an empty, misspelled `requriments.txt` file rather than a populated `requirements.txt`. Dependencies and dataset paths must be configured for your environment before attempting to run the code.

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

This work builds on [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [ControlNet](https://github.com/lllyasviel/ControlNet), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), and [PPSNet](https://github.com/yahskapar/PPSNet/tree/main). We thank their maintainers for their open-source contributions.
