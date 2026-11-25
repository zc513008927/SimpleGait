# SketchGait

**Sketch It Out: Exploring Label-Free Structural Cues for Multimodal Gait Recognition**

Chao Zhang*, Zhuang Zheng*, Ruixin Li, and Zhanyong Mei  
\*Equal contribution.  

[**Paper (ACM Digital Library)**](https://dl.acm.org/doi/10.1145/3767308.3835541) · [**DOI**](https://doi.org/10.1145/3767308.3835541) · [**arXiv**](https://arxiv.org/abs/2603.05537) · [**Code**](https://github.com/zc513008927/SketchGait) · [**Data & Checkpoints (Baidu Netdisk)**](https://pan.baidu.com/s/1DillSa4rzk8-1yh4j5H2xA?pwd=5478)

**Published in:** Proceedings of the 34th ACM International Conference on Multimedia (ACM Multimedia 2026 / MM '26).

**DOI:** [10.1145/3767308.3835541](https://doi.org/10.1145/3767308.3835541)

## Overview

We introduce **Sketch**, a label-free visual modality extracted from RGB images using edge detectors (including TEED). It preserves dense structural information such as limb articulations and self-occlusion contours, complementing semantic human parsing. **SketchGait** uses Sketch and Parsing with modality-specific streams and a lightweight early fusion branch. **SketchGait++** further incorporates Silhouette.

The released configurations here target the **SUSTech1K** benchmark and are built upon [OpenGait](https://github.com/ShiqiYu/OpenGait). The paper also reports experiments on **CCPG**.

## Paper Figures

The following slots are reserved for figures taken directly from the **published paper**. Export the relevant figures from your own paper, save them in the repository's `assets/` directory with the filenames below, and push the image files together with this README.

### Figure 1: Motivation and Visual Representations

![Figure 1: Sketch and complementary gait representations](assets/paper_fig1.png)

### Figure 2: SketchGait Framework

![Figure 2: Overview of the SketchGait framework](assets/paper_fig2.png)

### Figure 3: Qualitative Visualization

![Figure 3: Qualitative visualization and modality comparisons](assets/paper_fig3.png)

> **Figure setup:** `assets/paper_fig1.png`, `assets/paper_fig2.png`, and `assets/paper_fig3.png` are placeholders for image paths, **not files currently included in the repository**. Export the corresponding figures from the published article, use these filenames (or update the links), and commit the figures to make them visible on GitHub. Please follow the applicable ACM figure-reuse rights and license.

## Main Results

All values below are **Rank-1 recognition accuracy (%)**, transcribed from the previously released paper tables. **OA@R1** denotes overall Rank-1 on SUSTech1K; **Mean** denotes the average over the four CCPG evaluation conditions. These are **paper-reported results**, not a fresh rerun of the released checkpoints.

### SUSTech1K (Paper Table 2)

| Input modalities | Method | NM | BG | CL | CR | UM | UN | OC | NT | OA@R1 | OA@R5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Silhouette | DeepGaitV2 | 86.5 | 82.8 | 49.2 | 80.4 | 83.3 | 81.9 | 86.0 | 28.0 | 80.9 | 91.9 |
| Parsing (M2FP) | DeepGaitV2 | 93.1 | 87.9 | 40.9 | 88.8 | 87.7 | 88.7 | 93.5 | 47.5 | 87.5 | 95.1 |
| Sketch (TEED) | DeepGaitV2 | 94.5 | 90.1 | 56.4 | 90.5 | 88.1 | 89.3 | 97.3 | 51.5 | 89.6 | 96.3 |
| Sketch + Parsing | MultiGait++ | 96.6 | 92.9 | 53.7 | 92.7 | 92.9 | 92.0 | 98.4 | 53.1 | 91.9 | 97.1 |
| **Sketch + Parsing** | **SketchGait** | **97.1** | **94.2** | **60.3** | **93.5** | **93.9** | **93.7** | **98.8** | **55.1** | **92.9** | **97.3** |
| **Sketch + Silhouette + Parsing** | **SketchGait++** | **97.5** | **94.4** | **60.1** | **93.6** | **94.1** | **93.9** | **98.9** | **54.0** | **93.1** | **97.4** |

### CCPG (Paper Table 3)

| Input modalities | Method | CL | UP | DN | BG | Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Silhouette | DeepGaitV2 | 78.6 | 84.8 | 80.7 | 89.2 | 83.3 |
| Sketch (TEED) | DeepGaitV2 | 61.0 | 67.4 | 75.7 | 83.1 | 71.2 |
| Parsing | DeepGaitV2 | 68.8 | 73.6 | 80.1 | 87.8 | 77.6 |
| Sketch + Parsing (TEED) | MultiGait++ | 69.9 | 77.3 | 80.2 | 91.0 | 79.6 |
| **Sketch + Parsing (TEED)** | **SketchGait** | **75.3** | **81.1** | **84.5** | **92.8** | **83.4** |
| **Sketch + Silhouette + Parsing (TEED)** | **SketchGait++** | **82.2** | **87.0** | **86.8** | **94.5** | **87.6** |

### Fusion Ablation on SUSTech1K (Paper Table 4)

| Sketch + Parsing configuration | NM | CL | NT | OA@R-1 |
| --- | ---: | ---: | ---: | ---: |
| Dual-branch, no fusion | 97.0 | 60.5 | 53.7 | 92.7 |
| Concat-only, single branch | 93.5 | 42.4 | 51.3 | 88.7 |
| Mid-stage (Stage-3), Add | 97.3 | 59.1 | 54.4 | 92.8 |
| Early-stage (Stage-1), fused-only Add | 96.0 | 51.1 | 52.7 | 91.3 |
| **Early-stage (Stage-1), tri-branch Add (ours)** | **97.1** | **60.3** | **55.1** | **92.9** |

For the authoritative final version of all experimental results and figures, please refer to the [published ACM paper](https://dl.acm.org/doi/10.1145/3767308.3835541). The tables above reflect the previously released paper version and should be checked against the final ACM version before being treated as a definitive transcription.

## Repository Contents

| Path | Description |
| --- | --- |
| [`opengait/`](opengait/) | Model implementations and the OpenGait training/evaluation framework. |
| [`configs/sketchgait/`](configs/sketchgait/) | Configurations for SketchGait and SketchGaitPP. |
| [`datasets/SUSTech1K/`](datasets/SUSTech1K/) | SUSTech1K protocol and preprocessing scripts. |
| [`output/SUSTech1K/`](output/SUSTech1K/) | Model output directories, including the checkpoint locations. |

## Data and Pretrained Checkpoints

The data and checkpoints used in our experiments are shared through **Baidu Netdisk**:

- **Download:** [sketchgait (Baidu Netdisk)](https://pan.baidu.com/s/1DillSa4rzk8-1yh4j5H2xA?pwd=5478)
- **Extraction code:** `5478`

| File | Description |
| --- | --- |
| `sils_teed_pars_merged.zip` | Preprocessed, aligned **64 × 64** three-modality SUSTech1K sequences (Silhouette, TEED, Parsing); recommended for direct training and evaluation. |
| `SUSTech1K-RAW_parsing_to_silhouette.zip` | Silhouette images extracted from the original SUSTech1K RGB data. |
| `SUSTech1K-RAW_parsing.rar` | Human parsing images extracted from the original RGB data. |
| `SUSTech1K-TEED_blackBG.zip` | TEED edge images with black backgrounds extracted from the RGB data. |
| `SketchGait-50000.pt` | Pretrained SketchGait checkpoint. |
| `SketchGaitPP-50000.pt` | Pretrained SketchGaitPP checkpoint. |

The raw-resolution modalities are provided for researchers who wish to investigate alternative preprocessing methods. For the original SUSTech1K dataset and its access terms, please refer to the [official SUSTech1K project](https://lidargait.github.io).

## Quick Start: Evaluation with the Provided Data

**1. Prepare the data.** Download and extract `sils_teed_pars_merged.zip` to a local directory. The preprocessed data contains aligned three-channel sequences at **64 × 64**, ordered as **Silhouette, TEED, Parsing**.

**2. Update the configuration.** Edit `data_cfg.dataset_root` in the relevant YAML file to point to the extracted `sils_teed_pars_merged` directory:

- [SketchGait config](configs/sketchgait/sketchgait.yaml)
- [SketchGaitPP config](configs/sketchgait/sketchgaitpp.yaml)

For example:

```yaml
data_cfg:
  dataset_name: SUSTech1K
  dataset_root: /path/to/sils_teed_pars_merged
  dataset_partition: ./datasets/SUSTech1K/SUSTech1K.json
```

**3. Place the checkpoints.** Extract or copy the downloaded checkpoint files to the corresponding directories under the repository root:

```text
output/SUSTech1K/
├── SketchGait/SketchGait/checkpoints/
│   └── SketchGait-50000.pt
└── SketchGaitpp/SketchGaitPP/checkpoints/
    └── SketchGaitPP-50000.pt
```

The provided YAML files use `evaluator_cfg.restore_hint: 50000`. Preserve the filenames and match each checkpoint to its respective configuration.

**4. Run evaluation.** From the repository root, using two GPUs:

```bash
# SketchGait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgait.yaml --phase test

# SketchGaitPP
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgaitpp.yaml --phase test
```

Adjust `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, and the configuration settings to your hardware.

## Training

Use the same preprocessed dataset and update `data_cfg.dataset_root` as above. To train from scratch, the provided configs set `trainer_cfg.restore_hint: 0`.

```bash
# SketchGait
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgait.yaml --phase train

# SketchGaitPP
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 \
  opengait/main.py --cfgs ./configs/sketchgait/sketchgaitpp.yaml --phase train
```

## Optional: Preprocessing from Original-Resolution Modalities

If you prefer to reproduce or modify the preprocessing rather than use the provided `64 × 64` data:

1. Download and extract the original-resolution **Silhouette**, **Parsing**, and **TEED** archives listed above.
2. Use [`datasets/SUSTech1K/pretreatment_three_modality_alignment.py`](datasets/SUSTech1K/pretreatment_three_modality_alignment.py) to spatially align the three modalities and produce **64 × 64** sequence-level PKL files. Set the input/output paths for each modality according to your local directories.
3. Use [`datasets/SUSTech1K/sils_sketch_pars_merged`](datasets/SUSTech1K/sils_sketch_pars_merged) to concatenate the aligned modalities along the channel dimension. Update the paths at the beginning of that script before running it. Its output channel order is **Silhouette, TEED (Sketch), Parsing**.
4. Set the resulting merged-data directory as `data_cfg.dataset_root`, then run training or evaluation using the commands above.

The supplied preprocessing follows the approach used in our experiments. **We encourage further exploration of the original-resolution three-modality data**, including different resize/interpolation methods, spatial normalization strategies, and alternative preprocessing pipelines. The provided processing procedure is not intended to restrict these investigations.

## Acknowledgement

This repository is built upon [OpenGait](https://github.com/ShiqiYu/OpenGait). We thank the creators of the [SUSTech1K](https://lidargait.github.io) and [CCPG](https://github.com/BNU-IVC/CCPG) benchmarks, and the authors of the edge-detection and human-parsing methods used to prepare the representations.

## Citation

If you use SketchGait, SketchGait++, the provided code, checkpoints, or data processing resources, please cite the **published ACM Multimedia 2026 paper**:

```bibtex
@inproceedings{zhang2026sketchgait,
  title     = {Sketch It Out: Exploring Label-Free Structural Cues for Multimodal Gait Recognition},
  author    = {Zhang, Chao and Zheng, Zhuang and Li, Ruixin and Mei, Zhanyong},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3767308.3835541},
  url       = {https://doi.org/10.1145/3767308.3835541}
}
```

For the exact publisher-exported citation (including page numbers), use the **Cite** option on the [ACM Digital Library article page](https://dl.acm.org/doi/10.1145/3767308.3835541).
