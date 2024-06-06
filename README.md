# **DiSR-NeRF: Diffusion-Guided View-Consistent Super-Resolution NeRF**

[![Arxiv](https://img.shields.io/badge/arXiv-2404.00874-b31b1b.svg)](https://arxiv.org/abs/2404.00874)
[![YouTube](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=for-the-badge&logo=YouTube&logoColor=white)](https://www.youtube.com/watch?v=-zoEWBHGQoE&t=1s)

[Jie Long Lee](https://scholar.google.com/citations?user=vIgl6XQAAAAJ&hl=en) , [Chen Li](https://scholar.google.com.sg/citations?user=6_rJ2pcAAAAJ&hl=en), [Gim Hee Lee](https://scholar.google.ca/citations?user=7hNKrPsAAAAJ&hl=en)


## Abstract
We present DiSR-NeRF, a diffusion-guided framework for view-consistent super-resolution (SR) NeRF. Unlike prior works, we circumvent the requirement for high-resolution (HR) reference images by leveraging existing powerful 2D super-resolution models. Nonetheless, independent SR 2D images are often inconsistent across different views. We thus propose Iterative 3D Synchronization (I3DS) to mitigate the inconsistency problem via the inherent multi-view consistency property of NeRF. Specifically, our I3DS alternates between upscaling low-resolution (LR) rendered images with diffusion models, and updating the underlying 3D representation with standard NeRF training. We further introduce Renoised Score Distillation (RSD), a novel score-distillation objective for 2D image resolution. Our RSD combines features from ancestral sampling and Score Distillation Sampling (SDS) to generate sharp images that are also LR-consistent. Qualitative and quantitative results on both synthetic and real-world datasets demonstrate that our DiSR-NeRF can achieve better results on NeRF super-resolution compared with existing works. Code and video results available at the project website.

## Installation
```
git clone https://github.com/leejielong/DiSR-NeRF
cd DiSR-NeRF

conda create -n disrnerf
conda activate disrnerf

# Install packages
pip install -r requirements.txt
```

### Training
Download NeRF-Synthetic and LLFF datasets [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Create data directory as follows:
```
configs
data
├── blender
│   ├── chair
│   └── drums
└── nerf_llff_data
    ├── fern
    └── flower
```
```
python launch.py --config configs/nerfdiffusr-sr.yaml --train
```

### Testing
```
python launch.py --config configs/nerfdiffusr-sr.yaml --test
```

### Citations
```
@misc{lee2024disrnerf,
      title={DiSR-NeRF: Diffusion-Guided View-Consistent Super-Resolution NeRF}, 
      author={Jie Long Lee and Chen Li and Gim Hee Lee},
      year={2024},
      eprint={2404.00874},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Acknowledgement
This implementation is built upon [threestudio](https://github.com/threestudio-project/threestudio). We thank the authors for the contribution.
