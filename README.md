# SHARP-Net

**S**tage-adaptive **H**eterogeneity-**A**ware **R**emote-sensing **P**arsing **Net**work for very-high-resolution (VHR) aerial image semantic segmentation.

SHARP-Net is an *identity-preserving* parsing framework built on a Swin-Base encoder. It targets three recurring, **structured** sources of error in VHR aerial segmentation, and aligns one dedicated, near-identity-initialised module with each of them:

| Challenge | Module | What it does |
|-----------|--------|--------------|
| **C1** Optimal decoding scale varies spatially within a tile | **HG-SAF** — Heterogeneity-Guided Stage-Adaptive Fusion | Predicts *per-pixel* softmax weights over the four encoder stages, conditioned on a deep-feature heterogeneity map. Edges/small objects route to shallow high-resolution cues; homogeneous interiors route to deep semantics. |
| **C2** Repeated downsampling attenuates high-frequency detail | **FRA** — Frequency-Residual Adapter | A bounded, low-rank, identity-initialised residual FFT branch (`gate = 1 + α·tanh(·)`, output scaled by a learnable `γ` initialised to 0) that restores high-frequency content without destabilising the pretrained backbone. |
| **C3** Errors concentrate in a few confusable class pairs | **CATP** — Confusion-Aware Tri-Prior decoder | Three orthogonal auxiliary priors: a boundary prior, a small-object prior, and a prototype-contrast prior on pre-declared confusion pairs. |

A fixed refinement trunk (SSDB at shallow stages, BCSMamba at deep stages, plus an SOE small-object block) sits between the encoder and the three contribution modules. The trunk is a strong but **fixed** feature extractor and is *not* part of the claimed contributions.

> **Design principle — identity at initialisation.** Every new branch is the identity map at step 0 (zero-initialised gate / `γ = 0` residual / near-identity boundary attention). Each module therefore behaves as a *residual correction* to the pretrained features rather than a parallel branch learned from scratch, which keeps fine-tuning stable on relatively small VHR datasets.

---

## Results

All numbers are mean IoU (%) under one identical protocol (single-scale, **no test-time augmentation** for SHARP-Net). SHARP-Net entries are the mean over three random-seed runs. Per-class IoU and mean IoU are mutually consistent (mIoU is the exact mean of the per-class IoU values).

| Dataset | mIoU (%) | Standing |
|---------|:--------:|----------|
| ISPRS Vaihingen | 84.48 | strong 2nd (no TTA) |
| ISPRS Potsdam | 88.20 | **1st** |
| LoveDA (val) | 55.17 | close 2nd |
| OpenEarthMap (val) | 67.70 | **1st** |

**LoveDA per-class IoU (validation, EMA, verified from the training log):**

| Background | Building | Road | Water | Barren | Forest | Agricultural | **mIoU** |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 52.32 | 60.16 | 58.80 | 70.38 | **39.80** | 42.67 | 62.05 | **55.17** |

SHARP-Net delivers its clearest per-class gains on heterogeneous, low-texture rural classes (e.g. LoveDA *Barren*, well above competing methods) — precisely the classes the heterogeneity-guided fusion is designed for.

---

## Repository layout

```
SHARP-Net/
├── README.md                 # this file
├── sharpnet.py               # core SHARP-Net model (HG-SAF, FRA, CATP, trunk)
├── train_sharpnet_unified.py # unified training entry point
└── ...                       # dataset wrappers, losses, eval utilities
```

> Pretrained weights and datasets are **not** bundled in this repository (size / licensing). See the sections below for how to reproduce or where weights live.

---

## Installation

```bash
# Python 3.10+, CUDA-capable GPU recommended (training used a single RTX-4090)
pip install torch torchvision        # match your CUDA version
pip install timm numpy opencv-python pillow tqdm
# Mamba scan (for the deep-stage BCSMamba refinement)
pip install mamba-ssm
```

The backbone is the `timm` Swin-B checkpoint
`swin_base_patch4_window12_384.ms_in22k_ft_in1k` (ImageNet-22k pretrained, ImageNet-1k fine-tuned), downloaded automatically by `timm`.

---

## Usage

### Training

```bash
python train_sharpnet_unified.py --dataset loveda     --data_root /path/to/LoveDA
python train_sharpnet_unified.py --dataset potsdam    --data_root /path/to/Potsdam
python train_sharpnet_unified.py --dataset vaihingen  --data_root /path/to/Vaihingen
python train_sharpnet_unified.py --dataset oem        --data_root /path/to/OpenEarthMap
```

Key settings used in the paper: BFloat16 mixed precision (FRA's FFTs kept in FP32),
AdamW with differential learning rates (`lr_backbone = lr_head / 3`), cosine
annealing, batch size 8, gradient clipping at L2-norm 1, 80 epochs. LoveDA uses a
stronger "Route-B" recipe (CutMix + Mosaic, 3-epoch linear warmup, EMA shadow model
with decay 0.9995); the reported LoveDA number is the EMA mIoU.

### Evaluation

Each training run writes a `history.json` containing per-epoch `val_miou`,
`val_class_iou` (and their EMA variants) plus the best-epoch summary, and saves the
best checkpoint under `checkpoints/`.

---

## Pretrained weights

The four best checkpoints are produced by the training runs above and saved as:

| Dataset | Checkpoint |
|---------|------------|
| ISPRS Vaihingen | `outputs_vaihingen_sharpnet/checkpoints/SHARPNet_best.pth` (+ `SHARPNet_ema_best.pth`) |
| ISPRS Potsdam | `outputs_potsdam_sharpnet/checkpoints/SHARPNet_best.pth` |
| LoveDA | `outputs_loveda_sharpnet/checkpoints/SHARPNet_loveda_best.pth` |
| OpenEarthMap | `outputs_oem_sharpnet/checkpoints/SHARPNet_best.pth` |

> These files are large and are not committed here. They can be released as a
> separate download (release asset / cloud storage) on request.

---

## Datasets

| Dataset | Classes | GSD | Notes |
|---------|:-------:|-----|-------|
| [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 (5 scored) | 9 cm | IR-R-G |
| [ISPRS Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 (5 scored) | 5 cm | R-G-B |
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | 7 | 30 cm | urban + rural |
| [OpenEarthMap](https://open-earth-map.org/) | 8 | 25–50 cm | globally distributed |

On the two ISPRS benchmarks the five foreground classes are scored and the
*Clutter/background* class is excluded, following the common ISPRS convention.

---

## Citation

If you find SHARP-Net useful, please cite the accompanying paper (details to be
updated upon publication):

```bibtex
@article{sharpnet,
  title   = {SHARP-Net: Stage-Adaptive Heterogeneity-Aware Refinement for
             Very-High-Resolution Aerial Image Segmentation},
  author  = {Cao, Shuaishuai and Tang, Meng and Peng, Shuwei and Huang, Min},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released for academic research use. See `LICENSE` for details.
