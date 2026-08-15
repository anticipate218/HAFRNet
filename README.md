<div align="center">

# HAFR-Net

### Hierarchical Adaptive Feature Refinement Network<br>for VHR Remote Sensing Image Segmentation

Shuaishuai Cao, Meng Tang, Shuwei Peng, Xuan Liu, [Min Huang](mailto:huangm@jxnu.edu.cn)<sup>&#9993;</sup>,<br>
Jie Chen, Jiacheng Niu, Yong Chen, Edore Akpokodje, Hui Lin

<sub>Jiangxi Normal University &nbsp;&middot;&nbsp; Aberystwyth University &nbsp;&middot;&nbsp; Central South University</sub>

<p>
<a href="#citation"><img alt="Paper" src="https://img.shields.io/badge/paper-IEEE%20TGRS%20(under%20review)-00629B?style=flat-square"></a>
<a href="https://github.com/anticipate218/HAFRNet/releases/tag/weights-loveda-v1"><img alt="Weights" src="https://img.shields.io/badge/weights-LoveDA-2ea44f?style=flat-square"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?style=flat-square">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square"></a>
<img alt="Stars" src="https://img.shields.io/github/stars/anticipate218/HAFRNet?style=flat-square&color=ffd33d">
</p>

<img src="assets/qual_loveda.jpg" width="100%" alt="LoveDA qualitative comparison">

</div>

This repository is the official implementation of **HAFR-Net**, a Swin-Base framework for semantic
segmentation of very-high-resolution (VHR) aerial imagery. HAFR-Net *refines* a pretrained encoder
instead of replacing it: every added branch is initialized at, or close to, the identity map, so each
refinement stage acts as a bounded residual correction and fine-tuning stays stable on the relatively
small VHR benchmarks. It reaches **84.48 / 88.20 / 55.17 / 67.70** mIoU on ISPRS Vaihingen, ISPRS
Potsdam, LoveDA and OpenEarthMap.

## News

- **2026-08** &nbsp;LoveDA best checkpoint released &mdash; [`weights-loveda-v1`](https://github.com/anticipate218/HAFRNet/releases/tag/weights-loveda-v1) (val mIoU 55.17).
- **2026-08** &nbsp;Manuscript submitted to *IEEE Transactions on Geoscience and Remote Sensing*; this repository is now public.

## Highlights

- **Identity-preserving refinement.** Zero-initialized gates, a frequency gain that starts at exactly one, and a residual scale that starts at exactly zero, so the first forward pass reproduces the baseline and the pretrained backbone is corrected only where it helps.
- **Three modules, three error sources.** Each module targets one recurring failure mode of VHR segmentation rather than adding generic capacity.
- **Matched-protocol evidence.** One Swin-B encoder, identical splits, schedule, augmentation and single-scale inference across all reported rows, so every margin is attributable to the decoding path.
- **Consistent gains.** +0.55 to +1.84 pp mIoU over the strongest controlled decoder on four benchmarks, for 8.5 M extra parameters and 11% more GFLOPs than Swin-B + UPerNet.

## Method

| | Challenge | Module | Design |
|:--:|---|---|---|
| **C1** | The optimal decoding scale varies from pixel to pixel inside one tile | **HG-SAF**<br>Heterogeneity-Guided Stage-Adaptive Fusion | Per-pixel softmax weights over the four encoder stages, conditioned on a local feature-heterogeneity statistic. The gate is zero-initialized, so fusion starts as the plain mean of the projected stages. |
| **C2** | Repeated downsampling attenuates high-frequency detail | **FRA**<br>Frequency-Residual Adapter | A residual rFFT branch inside a channel bottleneck with a tanh-bounded per-coefficient gain (`g = 1 + alpha * tanh(.)`, `alpha = 0.5`) and a learnable residual scale initialized to zero, making the first forward pass an exact identity. |
| **C3** | Errors concentrate in a few confusable class pairs | **CATP**<br>Confusion-Aware Tri-Prior decoder | Boundary prior, objectness prior, and a prototype-contrast term on a relation set of class pairs frozen from a training-only pilot split. Pairwise confusion mass on those pairs drops by 23&ndash;28%. |

A **fixed** preparation trunk sits between the encoder and the three refinement stages: SSDB
(Spectral-Spatial Decoupled Block) at the two high-resolution stages, BCS-Mamba (Bi-directional
Cross-Scan Mamba) at the two low-resolution stages, and an SOE (Small Object Enhancement) block on
the fused path. It is measured separately in the paper rather than claimed as a contribution.

## Results

Matched protocol: one Swin-B encoder, identical splits, schedule and augmentation, single-scale
inference for every entry.

<div align="center">

| Dataset | Classes | mIoU (%) | + flip/rot TTA |
|---|:--:|:--:|:--:|
| ISPRS Vaihingen | 5 scored | 84.12 | **84.48** |
| ISPRS Potsdam | 5 scored | 87.86 | **88.20** |
| LoveDA (official val) | 7 | **55.17** | &mdash; |
| OpenEarthMap | 8 | **67.70** | &mdash; |

</div>

Published results for the same benchmarks are tabulated in the paper as literature context only,
since backbones and inference settings differ across sources.

## Model Zoo

<div align="center">

| Dataset | mIoU (%) | Checkpoint |
|---|:--:|:--:|
| LoveDA | 55.17 | [`HAFRNet_loveda_best.pth`](https://github.com/anticipate218/HAFRNet/releases/tag/weights-loveda-v1) (478 MB) |
| ISPRS Vaihingen | 84.48 | upon acceptance |
| ISPRS Potsdam | 88.20 | upon acceptance |
| OpenEarthMap | 67.70 | upon acceptance |

</div>

Checkpoints are attached to the [Releases](https://github.com/anticipate218/HAFRNet/releases) page and
are not tracked in git history.

```python
import torch

ck = torch.load("HAFRNet_loveda_best.pth", map_location="cpu")
print(ck["best_miou"], ck["best_is_ema"])    # 0.5517  True
model.load_state_dict(ck["model_state_dict"])
model.eval()
```

The archive stores `model_state_dict`, `best_miou`, `best_is_ema`, `epoch` and `history` (per-epoch
`val_miou`, `val_class_iou` and their EMA variants). The LoveDA entry holds the **EMA shadow
weights**, the setting the LoveDA number is reported under. Top-level keys of the state dict are
`_swin_full`, `ssdb_stages`, `mamba_stages`, `proj_pre`, `hg_saf`, `fra`, `soe`, `decoder` and
`aux_heads`; the `aux_heads` tensors are deep-supervision heads used during training only.

## Visualizations

<details open>
<summary><b>ISPRS Vaihingen and Potsdam</b></summary>
<br>
<img src="assets/qual_isprs.jpg" width="100%" alt="ISPRS qualitative comparison">
<sub>Boxes mark vehicles and boundaries between adjacent classes, the regions discussed in the HG-SAF and CATP analyses.</sub>
</details>

<details>
<summary><b>LoveDA</b></summary>
<br>
<img src="assets/qual_loveda.jpg" width="100%" alt="LoveDA qualitative comparison">
<sub>Boxes mark thin rural roads and low-texture <i>Barren</i> regions.</sub>
</details>

<details>
<summary><b>OpenEarthMap</b></summary>
<br>
<img src="assets/qual_oem.jpg" width="100%" alt="OpenEarthMap qualitative comparison">
<sub>Boxes mark <i>Rangeland</i>&nbsp;&harr;&nbsp;<i>Agriculture</i> transitions and thin roads, the pair with the largest confusion-mass reduction from CATP.</sub>
</details>

## Getting Started

### Installation

```bash
# Python 3.10+, one CUDA GPU (all reported runs used a single RTX 4090)
pip install torch torchvision        # match your CUDA build
pip install timm numpy opencv-python pillow tqdm
pip install mamba-ssm                # BCS-Mamba preparation stages
```

The backbone is the `timm` checkpoint `swin_base_patch4_window12_384.ms_in22k_ft_in1k`
(ImageNet-22k pretrained, ImageNet-1k fine-tuned), downloaded automatically by `timm`.

### Datasets

| Dataset | Classes | GSD | Bands | Download |
|---|:--:|:--:|:--:|---|
| ISPRS Vaihingen | 6 (5 scored) | 9 cm | IR-R-G | [ISPRS](https://www.isprs.org/education/benchmarks/UrbanSemLab/) |
| ISPRS Potsdam | 6 (5 scored) | 5 cm | R-G-B | [ISPRS](https://www.isprs.org/education/benchmarks/UrbanSemLab/) |
| LoveDA | 7 | 30 cm | R-G-B | [LoveDA](https://github.com/Junjue-Wang/LoveDA) |
| OpenEarthMap | 8 | 25&ndash;50 cm | R-G-B | [OpenEarthMap](https://open-earth-map.org/) |

On the two ISPRS benchmarks the five foreground classes are scored, *Clutter* is excluded, and the
official 3-px boundary erosion is applied.

### Training

```bash
python train_hafrnet_unified.py --dataset vaihingen --data_root /path/to/Vaihingen
python train_hafrnet_unified.py --dataset potsdam   --data_root /path/to/Potsdam
python train_hafrnet_unified.py --dataset loveda    --data_root /path/to/LoveDA
python train_hafrnet_unified.py --dataset oem       --data_root /path/to/OpenEarthMap
```

512&times;512 crops with train and test stride 256; at most 80 epochs at batch size 8; AdamW with head
learning rate 1.2e-4 and backbone learning rate 4e-5; cosine schedule with a 5-epoch warm-up; weight
decay 1e-4; gradient clipping at L2 norm 1; BF16 mixed precision with the spectral FFTs in FP32.
Augmentation is horizontal or vertical flip plus discrete 90&deg; rotation.

LoveDA adds a stronger photometric recipe on top: colour jitter 0.4, Gaussian blur 0.3, an EMA shadow
model with decay 0.999, and early stopping with patience 12; its reported number is the EMA mIoU.
Seeds are 42, 43 and 44, and the kept checkpoint is the one with the best validation mIoU.

### Evaluation

Each run writes `history.json` with per-epoch `val_miou` and `val_class_iou` (plus the EMA variants)
and saves the best checkpoint under `checkpoints/`.

## TODO

- [x] LoveDA best checkpoint
- [ ] Training and evaluation code &mdash; *upon acceptance*
- [ ] Vaihingen, Potsdam and OpenEarthMap checkpoints &mdash; *upon acceptance*
- [ ] Configuration files reproducing every table of the paper
- [ ] Architecture figure and module diagrams

## Citation

```bibtex
@article{cao2026hafrnet,
  title   = {Hierarchical Adaptive Feature Refinement Network for VHR Remote Sensing
             Image Segmentation},
  author  = {Cao, Shuaishuai and Tang, Meng and Peng, Shuwei and Liu, Xuan and Huang, Min
             and Chen, Jie and Niu, Jiacheng and Chen, Yong and Akpokodje, Edore and Lin, Hui},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgement

Built on [`timm`](https://github.com/huggingface/pytorch-image-models) for the Swin-B backbone, and on
the public ISPRS, [LoveDA](https://github.com/Junjue-Wang/LoveDA) and
[OpenEarthMap](https://open-earth-map.org/) benchmarks. We thank the teams that maintain these
datasets.

## License

Released under the [MIT License](LICENSE) for academic research use.

## Contact

Questions and issues are welcome. For anything beyond the issue tracker, contact
[huangm@jxnu.edu.cn](mailto:huangm@jxnu.edu.cn).
