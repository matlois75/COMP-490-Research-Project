# COMP 490 Research Project

Mathys Loiselle <br/>
Concordia University

## Overview

This project is an implementation and evaluation of the Difference Target Propagation (DTP) algorithm, a bio-plausible alternative to Backpropagation (BP) for training deep neural networks. The performance of a DTP-trained network is compared against a standard feedforward network trained with BP on image classification tasks.

The project uses Weights & Biases (`wandb`) for experiment tracking and logging.

## Project Structure

```
.
├── configs/                # Configuration files for training
│   ├── bp_configs.yaml
│   └── dtp_configs.yaml
├── data/                   # Datasets (auto-downloaded)
├── Difference_Target_Propagation/ # DTP model implementation
│   ├── dtp_layers.py
│   └── dtp_network.py
├── Vanilla_Backpropagation/ # BP model implementation
│   └── bp_network.py
├── train_bp.py             # Training script for BP model
├── train_dtp.py            # Training script for DTP model
├── test_dtp.py             # Testing script for DTP model
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Clone the repository:**

    ```sh
    git clone <your-repository-url>
    cd COMP-490-Research-Project
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```sh
    pip install -r requirements.txt
    ```
    Note: The [`requirements.txt`](requirements.txt) file specifies a CUDA-enabled version of PyTorch. Please adjust it according to your system's hardware (e.g., for a CPU-only version).

## Datasets

The following datasets are supported:

- MNIST
- CIFAR-10
- CIFAR-100

The datasets will be automatically downloaded to the `data/` directory the first time you run a training script.

## Usage

### Training

Model hyperparameters and training settings can be configured in the YAML files inside the [`configs/`](configs/) directory. You can also override these settings using command-line arguments.

#### Difference Target Propagation (DTP)

To train the DTP network, run the [`train_dtp.py`](train_dtp.py) script.

```sh
python train_dtp.py --dataset <dataset_name>
```

- `<dataset_name>` can be `mnist`, `cifar10`, or `cifar100`.

You can override configuration options from [`configs/dtp_configs.yaml`](configs/dtp_configs.yaml) via command-line arguments:

```sh
python train_dtp.py --dataset=mnist --epochs=100 --batch_size=256 --forward_lr=1e-4 etc...
```

#### Backpropagation (BP)

To train the standard backpropagation network, run the [`train_bp.py`](train_bp.py) script. The configuration is in [`configs/bp_configs.yaml`](configs/bp_configs.yaml).

```sh
python train_bp.py
```

### Testing

To evaluate a trained DTP model on the test set, use the [`test_dtp.py`](test_dtp.py) script.

```sh
python test_dtp.py --dataset <dataset_name>
```

This will automatically find and use the most recent checkpoint for the specified dataset. To use a specific checkpoint, provide the run directory name with the `--ckpt` argument:

```sh
# Example: python test_dtp.py --dataset mnist --ckpt run_dtp_mnist_1_jan_2024_1
python test_dtp.py --dataset <dataset_name> --ckpt <run_directory_name>
```

## Implementation notes on Difference Target Propagation (DTP)

This implementation is based on the 2015 paper by Lee et al.

### What is replicated from Lee et al. (2015)

- **Forward / inverse layer pairing**: Every hidden layer has `f_i` and `g_i` modules, with `tanh` activations as in the paper’s deterministic experiments.
- **Difference-target formula**: Targets are generated with the paper’s Eq. (15), implemented in `compute_targets` of `dtp_network.py`.
  \[
  \hat h*{i-1}=h*{i-1}+g_i(\hat h_i)-g_i(h_i)
  \]
- **Forward-loss + inverse-loss training**: Each step minimises top cross-entropy, layer-wise forward MSE, and inverse MSE, mirroring Algorithm 1 of the paper.
- **Gaussian corruption in the inverse loss**: Noise is injected with a fixed σ during inverse reconstruction, matching the denoising-style training in Sec. 2.2.

### What is different

| Aspect                       | This repo                                                                          | Original paper                               |
| :--------------------------- | :--------------------------------------------------------------------------------- | :------------------------------------------- |
| **Weight init**              | `N(0, 0.01²)` for every `nn.Linear`.                                               | Orthogonal matrices.                         |
| **Optimizer**                | Separate RMSprop for forward and inverse params, with CosineAnnealing LR schedule. | RMSprop with layer-specific static LR.       |
| **Inverse for output layer** | Not used (no `g_L`).                                                               | Present but sometimes omitted in later work. |
| **Noise schedule**           | Fixed σ from config.                                                               | σ decays over epochs (Eq. 27 discussion).    |
| **Warm-up per batch**        | `k_g_updates` inverse-only steps before each joint step.                           | Not described.                               |
| **Datasets**                 | MNIST, CIFAR-10/100 supported.                                                     | MNIST and CIFAR-10 only.                     |
| **Experiment tracking**      | Weights & Biases logging.                                                          | None.                                        |

### Reference

Dong-Hyun Lee, Saizheng Zhang, Asja Fischer and Yoshua Bengio, Difference Target Propagation, in Machine Learning and Knowledge Discovery in Databases, pages 498-515, Springer International Publishing, 2015
