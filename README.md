# HAFR-Net

**Hierarchical Adaptive Feature Refinement Network for VHR Remote Sensing Image Segmentation**

HAFR-Net is a Swin-Base segmentation framework for very-high-resolution (VHR) aerial
imagery. It follows one design principle: the pretrained encoder is *refined*, not
replaced. Every added branch starts at (or close to) the identity map, so each stage
acts as a bounded residual correction to the pretrained features rather than as a
parallel branch learned from scratch, which keeps fine-tuning stable on the
relatively small VHR datasets.

Three refinement stages are aligned one-to-one with three recurring, structured
sources of error:

| Challenge | Module | What it does |
|---|---|---|
| **C1** The best decoding scale varies from pixel to pixel inside one tile | **HG-SAF** — Heterogeneity-Guided Stage-Adaptive Fusion | Predicts per-pixel softmax weights over the four encoder stages, conditioned on a local feature-heterogeneity statistic. The gate is zero-initialized, so fusion starts as the plain mean of the projected stages. |
| **C2** Repeated downsampling attenuates high-frequency detail | **FRA** — Frequency-Residual Adapter | A residual rFFT branch inside a channel bottleneck, with a tanh-bounded per-coefficient gain (`g = 1 + alpha * tanh(...)`, `alpha = 0.5`) and a learnable residual scale initialized to zero, so the first forward pass is an exact identity. |
| **C3** Errors concentrate in a few confusable class pairs | **CATP** — Confusion-Aware Tri-Prior decoder | Three auxiliary structural signals: a boundary prior, an objectness prior, and a prototype-contrast term on a relation set of class pairs frozen from a training-only pilot split. |

A fixed preparation stage sits between the encoder and the three refinement stages:
SSDB (Spectral-Spatial Decoupled Block) at the two high-resolution stages,
BCS-Mamba (Bi-directional Cross-Scan Mamba) at the two low-resolution stages, and an
SOE (Small Object Enhancement) block on the fused path. The preparation stage is a
strong but **fixed** feature extractor and is measured separately rather than claimed
as a contribution.

---

## Results

Matched protocol: one Swin-B encoder, identical splits, schedule, augmentation and
single-scale inference for every entry, so a row difference is a difference in the
decoding path.

| Dataset | mIoU (%) | With flip+rotation TTA |
|---|:---:|:---:|
| ISPRS Vaihingen | 84.12 | 84.48 |
| ISPRS Potsdam | 87.86 | 88.20 |
| LoveDA (official val) | 55.17 | — |
| OpenEarthMap | 67.70 | — |

Published-result tables for the same benchmarks are reported in the paper as
literature context, since backbones and inference settings differ across sources.

---

## Pretrained weights

Weights are attached to the [Releases](https://github.com/anticipate218/HAFRNet/releases)
page, not tracked in the repository.

| Dataset | Checkpoint | val mIoU |
|---|---|:---:|
| LoveDA | [`HAFRNet_loveda_best.pth`](https://github.com/anticipate218/HAFRNet/releases/tag/weights-loveda-v1) | **55.17** |
| ISPRS Vaihingen | *coming soon* | — |
| ISPRS Potsdam | *coming soon* | — |
| OpenEarthMap | *coming soon* | — |

Loading:

```python
import torch

ck = torch.load("HAFRNet_loveda_best.pth", map_location="cpu")
print(ck["best_miou"], ck["best_is_ema"])   # 0.5517  True
model.load_state_dict(ck["model_state_dict"])
model.eval()
```

The archive stores five keys: `model_state_dict`, `best_miou`, `best_is_ema`,
`epoch`, and `history` (per-epoch `val_miou`, `val_class_iou` and their EMA
variants). The LoveDA entry is the **EMA shadow weights**, which is the setting the
LoveDA number is reported under.

Top-level keys of the state dict, if you want to load the stages individually:
`_swin_full`, `ssdb_stages`, `mamba_stages`, `proj_pre`, `hg_saf`, `fra`, `soe`,
`decoder`, `aux_heads`. The `aux_heads` entries are deep-supervision heads used only
during training.

---

## Installation

```bash
# Python 3.10+, CUDA-capable GPU (training used a single RTX 4090)
pip install torch torchvision        # match your CUDA version
pip install timm numpy opencv-python pillow tqdm
pip install mamba-ssm                # for the BCS-Mamba preparation stages
```

The backbone is the `timm` Swin-B checkpoint
`swin_base_patch4_window12_384.ms_in22k_ft_in1k` (ImageNet-22k pretrained,
ImageNet-1k fine-tuned), downloaded automatically by `timm`.

---

## Usage

### Training

```bash
python train_sharpnet_unified.py --dataset loveda    --data_root /path/to/LoveDA
python train_sharpnet_unified.py --dataset potsdam   --data_root /path/to/Potsdam
python train_sharpnet_unified.py --dataset vaihingen --data_root /path/to/Vaihingen
python train_sharpnet_unified.py --dataset oem       --data_root /path/to/OpenEarthMap
```

Settings used in the paper: 512x512 crops with train and test stride 256, at most 80
epochs at batch size 8, AdamW with head learning rate 1.2e-4 and backbone learning
rate 4e-5, a cosine schedule with a 5-epoch warm-up, weight decay 1e-4, gradient
clipping at L2 norm 1, and BF16 mixed precision with the spectral FFTs kept in FP32.
Augmentation is horizontal or vertical flip and discrete 90-degree rotation. LoveDA is
the only dataset with a stronger photometric recipe: colour jitter 0.4, Gaussian blur
0.3, an EMA shadow model with decay 0.999, and early stopping with patience 12; its
reported number is the EMA mIoU. Seeds are 42, 43 and 44, and the kept checkpoint is
the one with the best validation mIoU.

### Evaluation

Each run writes `history.json` with per-epoch `val_miou` and `val_class_iou` (plus
their EMA variants) and saves the best checkpoint under `checkpoints/`.

---

## Datasets

| Dataset | Classes | GSD | Notes |
|---|:---:|---|---|
| [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 (5 scored) | 9 cm | IR-R-G |
| [ISPRS Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 (5 scored) | 5 cm | R-G-B |
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | 7 | 30 cm | urban and rural |
| [OpenEarthMap](https://open-earth-map.org/) | 8 | 25-50 cm | globally distributed |

On the two ISPRS benchmarks the five foreground classes are scored, *Clutter* is
excluded, and the official 3-px boundary erosion is applied.

---

## Citation

```bibtex
@article{cao2026hafrnet,
  title   = {Hierarchical Adaptive Feature Refinement Network for VHR Remote Sensing Image Segmentation},
  author  = {Cao, Shuaishuai and Tang, Meng and Peng, Shuwei and Liu, Xuan and Huang, Min
             and Chen, Jie and Niu, Jiacheng and Chen, Yong and Akpokodje, Edore and Lin, Hui},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

Released for academic research use. See `LICENSE` for details.
