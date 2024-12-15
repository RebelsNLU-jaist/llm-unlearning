# AAAI-25: On Effects of Steering Latent Representations for Large Language Model Unlearning

This repository contains code and resources for the paper **"On Effects of Steering Latent Representations for Large Language Model Unlearning"** by Dang Huu-Tien, Trung-Tin Pham, Hoang Thanh-Tung, and Naoya Inoue.

## Table of Contents

1. [Setup](#1-setup)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Unlearning](#3-unlearning)
4. [Evaluation](#4-evaluation)
5. [Reference](#5-reference)

## 1. Setup

Follow the steps below to set up the environment:

```bash
conda create -n unlearning
conda activate unlearning
pip install -r requirements.txt
```

## 2. Dataset Preparation

Create a directory to store datasets:

```bash
mkdir data/
```

Download the required datasets from the [WMDP repository](https://github.com/centerforaisafety/wmdp) and place them in the `data/` directory.

## 3. Unlearning

Run the unlearning process with the following command:

```bash
python -m baselines.adap_rmu.unlearn \
    --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
    --max_num_batches 500 \
    --alpha 1200,1200 \
    --batch_size 4 \
    --seed 42 \
    --scale 5.0 \
    --layer_id 7 \
    --layer_ids 5,6,7 \
    --verbose
```

Alternatively, perform a grid search for hyperparameter tuning by running:

```bash
bash experiments/adap_rmu.sh
```

## 4. Evaluation

We use the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework for evaluation.
```
!lm-eval --model hf \
    --model_args pretrained="checkpoints/rmu/adaptive_HuggingFaceH4/zephyr-7b-beta_alpha-1200-1200_coeffs-6.5-6.5_batches-500_layer-7_scale-5" \
    --tasks mmlu,wmdp \
    --batch_size=16
```

## 5. Reference
If you find this work helpful, please cite our paper:

```bibtex
@article{huu2024effects,
  title={On Effects of Steering Latent Representations for Large Language Model Unlearning},
  author={Dang, Huu-Tien and Pham, Trung-Tin and Hoang, Thanh-Tung and Inoue, Naoya},
  journal={arXiv preprint arXiv:2408.06223},
  year={2024}
}
```
## Acknowledgment

This repository builds upon and extends the code from the [WMDP repository](https://github.com/centerforaisafety/wmdp).
