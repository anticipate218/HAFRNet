<div align="center">

<h1>HAFR-Net</h1>

<h3>Hierarchical Adaptive Feature Refinement Network<br>for VHR Remote Sensing Image Segmentation</h3>

<p>
<a href="https://www.grss-ieee.org/publications/tgrs/"><img alt="IEEE TGRS" src="https://img.shields.io/badge/IEEE%20TGRS-under%20review-00629B?style=flat-square"></a>
<a href="https://github.com/anticipate218/HAFRNet/releases/tag/weights-loveda-v1"><img alt="LoveDA weights" src="https://img.shields.io/badge/weights-LoveDA%20released-2ea44f?style=flat-square"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-BF16%20%7C%20CUDA-EE4C2C?style=flat-square">
<img alt="Backbone" src="https://img.shields.io/badge/backbone-Swin--B-8A2BE2?style=flat-square">
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square"></a>
</p>

<img src="assets/qual_loveda.jpg" width="100%" alt="LoveDA qualitative comparison">

<sub><b>LoveDA.</b> Boxes mark thin rural roads and low-texture <i>Barren</i> regions &mdash; the two places where every baseline in this row loses shape. Rightmost column is ours.</sub>

</div>

---

## Hello

I am HAFR-Net, and I am not a new backbone.

I start from a pretrained Swin-B that already sees this imagery quite well, and my whole job is to
fix the three things it keeps getting wrong on very-high-resolution aerial tiles. Every branch I
add begins **at the identity map**: my gates are zero-initialized, my frequency gain starts at
exactly one, my residual scale starts at exactly zero. On day one of fine-tuning I am, numerically,
the plain baseline &mdash; and then I improve from there. That is the whole design principle:

> **refine the pretrained encoder, do not replace it.**

It is also why I stay stable on datasets with a few hundred tiles, where branches learned from
scratch tend to fight the backbone instead of helping it.

---

## The three things I fix

<table>
<thead>
<tr><th width="20%">The problem</th><th width="24%">My module</th><th>How it works</th></tr>
</thead>
<tbody>
<tr>
<td><b>C1</b><br>The best decoding scale changes from pixel to pixel inside a single tile.</td>
<td><b>HG-SAF</b><br><sub>Heterogeneity-Guided<br>Stage-Adaptive Fusion</sub><br><br><i>"Let each pixel pick its own scale."</i></td>
<td>Predicts per-pixel softmax weights over the four encoder stages, conditioned on a local feature-heterogeneity statistic rather than on the features alone. The gate is zero-initialized, so fusion begins as the plain mean of the projected stages and only becomes selective where selectivity pays.</td>
</tr>
<tr>
<td><b>C2</b><br>Repeated downsampling quietly attenuates high-frequency detail.</td>
<td><b>FRA</b><br><sub>Frequency-Residual<br>Adapter</sub><br><br><i>"Give the thin structures their contrast back."</i></td>
<td>A residual rFFT branch inside a channel bottleneck with a tanh-bounded per-coefficient gain, <code>g = 1 + alpha * tanh(&middot;)</code> with <code>alpha = 0.5</code>, and a learnable residual scale initialized to zero. The first forward pass is therefore an exact identity, and the gain can never blow a frequency band up or erase it.</td>
</tr>
<tr>
<td><b>C3</b><br>Errors are not spread evenly &mdash; they pile up in a few confusable class pairs.</td>
<td><b>CATP</b><br><sub>Confusion-Aware<br>Tri-Prior decoder</sub><br><br><i>"Push apart the pairs that actually collide."</i></td>
<td>Three auxiliary structural signals: a boundary prior, an objectness prior, and a prototype-contrast term on a relation set of class pairs frozen ahead of time from a training-only pilot split. Pairwise confusion mass on those declared pairs drops by 23&ndash;28%.</td>
</tr>
</tbody>
</table>

Between the encoder and those three stages sits a **fixed** preparation trunk: SSDB (Spectral-Spatial
Decoupled Block) at the two high-resolution stages, BCS-Mamba (Bi-directional Cross-Scan Mamba) at
the two low-resolution stages, and an SOE (Small Object Enhancement) block on the fused path. It is a
strong feature extractor, but it is not mine to claim &mdash; the paper measures it separately, in its
own table, so you can see exactly how much of the margin it contributes.

---

## Results

One Swin-B encoder, identical splits, schedule and augmentation, single-scale inference for every
entry. A row difference is therefore a difference in the decoding path and nothing else.

<div align="center">

| Dataset | mIoU (%) | + flip/rotation TTA |
|---|:---:|:---:|
| ISPRS Vaihingen | 84.12 | **84.48** |
| ISPRS Potsdam | 87.86 | **88.20** |
| LoveDA (official val) | **55.17** | &mdash; |
| OpenEarthMap | **67.70** | &mdash; |

</div>

Published numbers from the literature are also tabulated in the paper, but as context only: backbones
and inference settings differ across sources, so those tables are not the evidence the claims rest on.

---

## Gallery

<details open>
<summary><b>ISPRS Vaihingen and Potsdam</b> &mdash; vehicles and adjacent class boundaries</summary>
<br>
<img src="assets/qual_isprs.jpg" width="100%" alt="ISPRS qualitative comparison">
<sub>Boxes mark vehicles and the boundaries between adjacent classes, the regions discussed in the HG-SAF and CATP analyses.</sub>
</details>

<details>
<summary><b>LoveDA</b> &mdash; thin rural roads and low-texture <i>Barren</i></summary>
<br>
<img src="assets/qual_loveda.jpg" width="100%" alt="LoveDA qualitative comparison">
<sub>Boxes mark thin rural roads and low-texture <i>Barren</i> regions.</sub>
</details>

<details>
<summary><b>OpenEarthMap</b> &mdash; <i>Rangeland</i> vs <i>Agriculture</i>, and thin roads</summary>
<br>
<img src="assets/qual_oem.jpg" width="100%" alt="OpenEarthMap qualitative comparison">
<sub>Boxes mark <i>Rangeland</i>&nbsp;&harr;&nbsp;<i>Agriculture</i> transitions and thin roads &mdash; the pair with the largest confusion-mass reduction from CATP.</sub>
</details>

---

## Weights

The LoveDA best checkpoint is **already public**. The rest follow when the paper is accepted.

<div align="center">

| Dataset | Checkpoint | val mIoU | Status |
|---|---|:---:|:---:|
| LoveDA | [`HAFRNet_loveda_best.pth`](https://github.com/anticipate218/HAFRNet/releases/tag/weights-loveda-v1) | **55.17** | available now |
| ISPRS Vaihingen | &mdash; | 84.48 | on acceptance |
| ISPRS Potsdam | &mdash; | 88.20 | on acceptance |
| OpenEarthMap | &mdash; | 67.70 | on acceptance |

</div>

Weights live on the [Releases](https://github.com/anticipate218/HAFRNet/releases) page, never in the
git history.

```python
import torch

ck = torch.load("HAFRNet_loveda_best.pth", map_location="cpu")
print(ck["best_miou"], ck["best_is_ema"])    # 0.5517  True
model.load_state_dict(ck["model_state_dict"])
model.eval()
```

The archive stores five keys: `model_state_dict`, `best_miou`, `best_is_ema`, `epoch` and `history`
(per-epoch `val_miou`, `val_class_iou` and their EMA variants). The LoveDA entry holds the **EMA
shadow weights**, which is the setting the LoveDA number is reported under.

If you would rather load the stages one at a time, the top level of the state dict is
`_swin_full`, `ssdb_stages`, `mamba_stages`, `proj_pre`, `hg_saf`, `fra`, `soe`, `decoder`,
`aux_heads` &mdash; where `aux_heads` are deep-supervision heads used during training only.

---

## Roadmap

- [x] LoveDA best checkpoint released
- [ ] Full training and evaluation code &mdash; **on acceptance**
- [ ] Vaihingen, Potsdam and OpenEarthMap checkpoints &mdash; **on acceptance**
- [ ] Config files reproducing every table in the paper

Until then, the sections below document the exact recipe behind the released weights, so the setup is
reproducible from the paper alone.

---

## Setup

<details>
<summary><b>Installation</b></summary>
<br>

```bash
# Python 3.10+, one CUDA GPU (all reported runs used a single RTX 4090)
pip install torch torchvision        # match your CUDA build
pip install timm numpy opencv-python pillow tqdm
pip install mamba-ssm                # BCS-Mamba preparation stages
```

The backbone is the `timm` checkpoint `swin_base_patch4_window12_384.ms_in22k_ft_in1k`
(ImageNet-22k pretrained, ImageNet-1k fine-tuned), which `timm` downloads for you.

</details>

<details>
<summary><b>Training recipe</b></summary>
<br>

```bash
python train_hafrnet_unified.py --dataset loveda    --data_root /path/to/LoveDA
python train_hafrnet_unified.py --dataset potsdam   --data_root /path/to/Potsdam
python train_hafrnet_unified.py --dataset vaihingen --data_root /path/to/Vaihingen
python train_hafrnet_unified.py --dataset oem       --data_root /path/to/OpenEarthMap
```

512&times;512 crops with train and test stride 256; at most 80 epochs at batch size 8; AdamW with head
learning rate 1.2e-4 and backbone learning rate 4e-5; cosine schedule with a 5-epoch warm-up; weight
decay 1e-4; gradient clipping at L2 norm 1; BF16 mixed precision with the spectral FFTs kept in FP32.
Augmentation is horizontal or vertical flip plus discrete 90&deg; rotation.

LoveDA is the one dataset with a stronger photometric recipe on top of that: colour jitter 0.4,
Gaussian blur 0.3, an EMA shadow model with decay 0.999, and early stopping with patience 12. Its
reported number is the EMA mIoU. Seeds are 42, 43 and 44, and the checkpoint kept is the one with the
best validation mIoU.

</details>

<details>
<summary><b>Evaluation</b></summary>
<br>

Each run writes `history.json` with per-epoch `val_miou` and `val_class_iou` (plus the EMA variants)
and saves the best checkpoint under `checkpoints/`. On the two ISPRS benchmarks the five foreground
classes are scored, *Clutter* is excluded, and the official 3-px boundary erosion is applied.

</details>

---

## Datasets

| Dataset | Classes | GSD | Notes |
|---|:---:|---|---|
| [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 (5 scored) | 9 cm | IR-R-G |
| [ISPRS Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/) | 6 (5 scored) | 5 cm | R-G-B |
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | 7 | 30 cm | urban and rural domains |
| [OpenEarthMap](https://open-earth-map.org/) | 8 | 25&ndash;50 cm | globally distributed |

---

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

---

## Acknowledgements

Built on [`timm`](https://github.com/huggingface/pytorch-image-models) for the Swin-B backbone and on
the public ISPRS, [LoveDA](https://github.com/Junjue-Wang/LoveDA) and
[OpenEarthMap](https://open-earth-map.org/) benchmarks. Thanks to the teams that keep those datasets
open &mdash; none of this exists without them.

Released under the [MIT License](LICENSE) for research use. Questions and issues are welcome.
