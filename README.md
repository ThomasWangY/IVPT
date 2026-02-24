# IVPT: Interpretable Visual Prompt Tuning based on Part Prototypes (ICLR 2026)

> **[Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](https://openreview.net/pdf?id=NHP2Y8IVMU)**
>
> [Yubin Wang](https://scholar.google.com/citations?user=mLeYNLoAAAAJ), [Xinyang Jiang](https://scholar.google.com/citations?user=JiTfWVMAAAAJ), [De Cheng](https://scholar.google.com/citations?user=180lASkAAAAJ),  Xiangqian Zhao,  [Zilong Wang](https://scholar.google.com/citations?user=gOaxHvMAAAAJ), [Dongsheng Li](https://scholar.google.com/citations?user=VNg5rA8AAAAJ), [Cairong Zhao](https://scholar.google.com/citations?user=z-XzWZcAAAAJ)


## ✨ Highlights

![main figure](docs/framework.png)

> **<p align="justify"> Abstract:** *Visual prompt tuning offers significant advantages for adapting pre-trained visual foundation models to specific tasks. However, current research provides limited insight into the interpretability of this approach, which is essential for enhancing AI reliability and enabling AI-driven knowledge discovery. In this paper, rather than learning abstract prompt embeddings, we propose a set of interpretable prompts within a part-prototype explanatory scheme. Each prompt is associated with a specific, human-understandable semantic concept that directly corresponds to a particular part of the image, making the model's behavior more transparent and explainable. Specifically, we present Interpretable Visual Prompt Tuning (IVPT), the first framework to explore interpretability in visual prompt tuning using part prototypes. We introduce a novel hierarchical structure of part prototypes to explain the learned prompts at various network layers. These category-agnostic prototypes are leveraged to discover concept regions, from where we aggregate features to obtain interpretable prompts integrated for fine-tuning. We perform comprehensive qualitative and quantitative evaluations on fine-grained classification benchmarks following the part-prototype explanatory scheme to show superior interpretability and accuracy.* </p>

## :rocket: Contributions

- We propose a novel framework for interpretable visual prompt tuning that uses part prototypes as a bridge to connect learnable prompts with human-understandable visual concepts.
- We introduce a hierarchical structure of part prototypes to explain prompts at multiple network layers while modeling their relationships in a fine-to-coarse alignment.
- We demonstrate the effectiveness of our approach through extensive qualitative and quantitative evaluations on fine-grained classification benchmarks. The results show improved interpretability and accuracy compared to both conventional visual prompt tuning methods and previous part-prototype-based methods.

## 📁 Project Structure

```
IVPT/
├── train_net.py                    # Main training & eval entry point
├── argument_parser_train.py        # Command-line argument parser
├── configs/                        # YAML configuration files
│   └── cub_default.yaml            #   Default config for CUB-200-2011
├── models/                         # Model architectures
│   ├── layers/                     #   Custom transformer layers
│   │   ├── transformer_layers.py   #     Attention / Block with QKV return
│   │   └── independent_mlp.py      #     Per-part MLP classifier
│   ├── individual_landmark_vit.py  #   Core IVPT ViT model
│   └── builder.py                  #   Model construction utilities
├── data_sets/                      # Dataset and data loading
│   ├── fg_bird_dataset.py          #   CUB / NABirds dataset
│   └── builder.py                  #   Dataset construction utilities
├── engine/                         # Training, evaluation, and losses
│   ├── distributed_trainer_ivpt.py #   Distributed trainer (DDP)
│   ├── eval_interpretability_nmi_ari_keypoint.py  # NMI/ARI/KPR eval
│   ├── eval_fg_bg.py               #   Foreground/Background IoU eval
│   └── losses/                     #   Loss functions
│       ├── builder.py              #     Loss construction
│       ├── consistency_loss.py     #     Cross-layer consistency
│       ├── equivarance_loss.py     #     Equivariance loss
│       ├── orthogonality_loss.py   #     Prototype orthogonality
│       ├── presence_loss.py        #     Presence loss (multiple variants)
│       ├── enforced_presence_loss.py  # Enforced presence
│       ├── pixel_wise_entropy_loss.py # Pixel-wise entropy
│       └── total_variation.py      #     Total variation
├── eval/                           # Interpretability evaluation scripts
│   ├── evaluate_consistency.py     #   Consistency & stability evaluation
│   ├── evaluate_parts.py           #   Part interpretability evaluation
│   └── ...                         #   Auxiliary eval utilities
├── utils/                          # Utility functions
│   ├── data_utils/                 #   Data transforms, affine transforms, samplers
│   ├── training_utils/             #   Optimizer, scheduler, DDP, checkpointing
│   ├── visualize_att_maps.py       #   Attention map overlay & hierarchy vis
│   ├── misc_utils.py               #   Attention computation, rollout, etc.
│   ├── get_landmark_coordinates.py #   Landmark coordinate extraction
│   ├── wandb_params.py             #   W&B logging utilities
│   ├── crop.py                     #   CUB bounding-box cropping
│   └── img_aug.py                  #   Augmentor-based data augmentation
├── scripts/                        # Shell scripts
│   ├── run_train.sh                #   Multi-GPU training launcher
│   ├── run_test.sh                 #   Classification evaluation
│   └── run_eval.sh                 #   Interpretability evaluation
├── docs/                           # Documentation & images
│   └── INSTRUCTION.md              #   Detailed training instructions
├── requirements.txt                # Python dependencies
└── environment.yml                 # Conda environment
```

## 🛠️ Installation 

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ivpt
pip install -r requirements.txt
```

### Option 2: pip

```bash
pip install -r requirements.txt
```

## 🗂️ Data Preparation

1. Download the CUB-200-2011 dataset from [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).
2. Unpack `CUB_200_2011.tgz` to the `datasets/` directory:

```
datasets/
└── CUB_200_2011/
    ├── images/
    ├── image_class_labels.txt
    ├── train_test_split.txt
    └── ...
```

3. **(Optional)** Run data preprocessing for cropping and augmentation:

```bash
python utils/crop.py
python utils/img_aug.py --data_path datasets/cub200_cropped
```

## 🧪 Training

### Quick Start

```bash
# Multi-GPU training (4 GPUs)
bash scripts/run_train.sh

# Or run directly
torchrun --nproc_per_node=4 train_net.py \
    --model_arch vit_base_patch14_reg4_dinov2.lvd142m \
    --pretrained_start_weights \
    --data_path datasets/CUB_200_2011 \
    --dataset cub \
    --batch_size 4 --epochs 25 \
    --freeze_backbone --gumbel_softmax \
    --n_pro 17,14,11,8,5
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_pro` | `17,14,11,8,5` | Prototype counts at different ViT layers (comma-separated) |
| `--model_arch` | `vit_base_patch14_reg4_dinov2.lvd142m` | Backbone architecture (timm model name) |
| `--modulation_type` | `layer_norm` | Prototype modulation type: `layer_norm`, `original`, `parallel_mlp`, `none` |
| `--freeze_backbone` | `False` | Freeze backbone parameters (recommended) |
| `--gumbel_softmax` | `False` | Use Gumbel-Softmax on attention maps |
| `--image_size` | `518` | Input image resolution |

All training parameters are also documented in the YAML config file `configs/cub_default.yaml` for reference.

See `scripts/run_train.sh` for a complete example, or refer to `docs/INSTRUCTION.md` for detailed training instructions (batch size scaling, single-GPU setup, etc.).

## 📊 Evaluation

### Classification Evaluation

```bash
# Use the convenience script
bash scripts/run_test.sh

# Or run directly (add --eval_only flag)
torchrun --nproc_per_node=4 train_net.py \
    --eval_only \
    --snapshot_dir ./snapshot \
    ... (same args as training)
```

### Interpretability Evaluation

Evaluate model interpretability using keypoint regression (KPR), NMI, ARI, or FG/BG IoU:

```bash
# Use the convenience script
bash scripts/run_eval.sh

# Or run directly
python eval/evaluate_consistency.py \
    --model_path ./snapshot/snapshot_best.pt \
    --dataset cub \
    --eval_mode nmi_ari \
    --num_parts 4 \
    --model_arch vit_base_patch14_reg4_dinov2.lvd142m \
    --data_path datasets/cub200_cropped \
    --n_pro 17,14,11,8,5
```

Supported `--eval_mode` values: `nmi_ari` | `kpr` | `fg_bg_iou`

## :art: Visualization

### Attention Map Visualization

Attention map overlays are **automatically saved** during evaluation runs. The visualizations show per-prototype region segmentation overlaid on input images, along with cropped patches for each prototype.

### Hierarchical Prototype Visualization

To generate hierarchical prototype visualizations showing cross-layer prototype relationships, add the `--enable_hierarchy_vis` flag:

```bash
torchrun --nproc_per_node=4 train_net.py \
    --eval_only \
    --enable_hierarchy_vis \
    --snapshot_dir ./snapshot \
    ... (same args as training)
```

This generates two types of output in the snapshot directory:

- **`results_hie_*/`**: Hierarchical folder layout organizing prototypes by their cross-layer relationships. Each sub-folder contains image crops illustrating the visual concept associated with a specific prototype at a particular layer.

<p align="center">
  <img src="docs/0_0.png" alt="Figure 1" width="20%">
  <img src="docs/0_20.png" alt="Figure 2" width="20%">
  <img src="docs/0_30.png" alt="Figure 3" width="20%">
</p>

- **`results_vis_*/`**: Multi-layer comparison views showing how part region segmentation evolves across different network layers.

<p align="center">
  <img src="docs/vis.png" alt="Visualization" width="95%">
</p>

## 🔍 Citation

If you use our work, please consider citing:

```bibtex
@misc{wang2026exploringinterpretabilityvisualprompt,
      title={Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts}, 
      author={Yubin Wang and Xinyang Jiang and De Cheng and Xiangqian Zhao and Zilong Wang and Dongsheng Li and Cairong Zhao},
      year={2026},
      eprint={2503.06084},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06084}, 
}
```

## 📧 Contact

If you have any questions, please create an issue on this repository or contact us at wangyubin2018@tongji.edu.cn.

## 😃 Acknowledgments

Our code is based on [PDiscoFormer](https://github.com/ananthu-aniraj/pdiscoformer) repository. We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.
