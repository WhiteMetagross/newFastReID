# newFastReID: A Modernized fastreid library implementation for General Instance Re-identification.

[](https://www.python.org/downloads/release/python-3110/)
[](https://pytorch.org/)
[](https://opensource.org/licenses/Apache-2.0)

This repository contains `newFastReID`, an enhanced and modernized fork of the original [FastReID](https://github.com/JDAI-CV/fast-reid) library. This version has been specifically updated and validated to ensure seamless compatibility with contemporary deep learning environments, including **Python 3.11** and **CUDA 12.1**.

`newFastReID` is a comprehensive, high-performance research platform built on PyTorch for a wide range of instance re-identification (ReID) tasks. It is engineered for flexibility, speed, and accessibility, catering to the needs of researchers and practitioners in person, vehicle, and general object re-identification.

-----

## Validation Environment:

The stability and functionality of this library have been verified in a specific, modern development environment. This ensures that users can replicate a known-good setup for immediate productivity.

  - **Operating System**: Windows 11
  - **Python Version**: 3.11
  - **NVIDIA CUDA Version**: 12.1
  - **PyTorch Version**: 2.0+ (with CUDA support)

Extensive testing has been performed to guarantee compatibility with models from the official [FastReID Model Zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md). The following models have been successfully validated in the environment above:

  - **VeRiWild (Vehicle ReID)**: Bag of Tricks (BOT) model with a ResNet-50 IBN-a backbone (`veriwild_bot_R50-ibn.pth`).
  - **Market-1501 (Person ReID)**: Shake Both Sides (SBS) model with a ResNet-101 IBN-a backbone (`market_sbs_R101-ibn.pth`).

-----

## Core Functionalities:

This library provides a robust suite of tools for ReID research and application development.

  - **Modern Environment Support**: Full compatibility with Python 3.11 and CUDA 12.1 allows for development with the latest tools and libraries.
  - **State of the Art Models**: Implementations of highly effective ReID architectures are provided, including Bag of Tricks (BoT), Shake Both Sides (SBS), and others.
  - **Multi-Dataset Support**: Built-in data loaders and configurations for standard ReID benchmarks such as Market-1501, DukeMTMC-reID, MSMT17, and VeRiWild facilitate standardized evaluation.
  - **Powerful Command Line Interface**: A comprehensive set of command line tools enables efficient training, testing, and inference without requiring direct modification of the source code.
  - **Guaranteed Model Zoo Compatibility**: A critical utility is included to analyze any pre-trained model from the original FastReID zoo and generate a perfectly matched YAML configuration file, ensuring seamless integration and reproducibility.

-----

## Installation Guide:

This section provides a step by step process for installing `newFastReID` and its dependencies.

### Prerequisites:

Before proceeding with the installation, the following components must be present on your system:

  - **Python**: Version 3.8 or higher is required. The library has been officially tested with Python 3.11.
  - **PyTorch**: Version 1.9 or higher is required. The library has been officially tested with PyTorch 2.0+. It is essential that PyTorch is installed with CUDA support to enable GPU acceleration. Instructions can be found on the [official PyTorch website](https://pytorch.org/get-started/locally/).
  - **NVIDIA CUDA Toolkit**: Version 11.0 or higher is recommended for GPU support. The library was tested with CUDA 12.1.
  - **Git**: Required for cloning the repository.

### Installation from Source:

As `newFastReID` is not yet available on the Python Package Index (PyPI), it must be installed directly from its GitHub repository.

1.  **Clone the Repository**
    Open a terminal or command prompt and use `git` to create a local copy of the repository:

    ```bash
    git clone https://github.com/WhiteMetagross/newFastReID.git
    ```

2.  **Navigate to the Project Directory**
    Change your current directory to the root of the newly cloned `newFastReID` folder:

    ```bash
    cd newFastReID
    ```

3.  **Install in Editable Mode**
    It is recommended to install the package in "editable" mode using `pip`. This method creates a symbolic link from your Python environment's `site-packages` directory to the source code, which means any changes you make will be immediately effective without needing to reinstall. Execute the installation command from the root of the project directory:

    ```bash
    pip install -e .
    ```

-----

## Setup and Usage Guide:

This library is primarily operated through YAML configuration files and a set of intuitive command line tools. The following guide details the end-to-end workflow from data preparation to inference.

Some usuage examples:
[Vehicles Tracking from an aerial drone view, with newFastReID](https://github.com/WhiteMetagross/ProjectIAV/tree/main/VehiclePathBoTSORTTrackerV2)
[Vehicles Tracking with Instance Segmentations, from an aerial drone view, with newFastReID]

### Step 1: Dataset Preparation

1.  Download the desired ReID datasets (e.g., Market-1501, DukeMTMC-reID, VeRiWild).
2.  Organize these datasets into a single root directory. The library expects a specific subdirectory structure for each dataset. An example of the required layout is shown below:
    ```
    /path/to/your/datasets/
    ├── market1501/
    │   ├── bounding_box_test/
    │   ├── bounding_box_train/
    │   ├── query/
    │   └── ...
    ├── dukemtmc/
    │   └── ...
    └── veriwild/
        └── ...
    ```

### Step 2: Experiment Configuration

1.  Navigate to the `fastreid/configs/` directory within the project structure.
2.  Select a base configuration file that corresponds to your target model and dataset (e.g., `fastreid/configs/Market1501/bagtricks_R50.yml`).
3.  Open this YAML file in a text editor. The most critical parameter to modify is `DATASETS.ROOT_DIR`. Set its value to the absolute path of the datasets directory created in the previous step.

### Step 3: Model Training

To initiate a training session, the `fastreid-train` command line tool is used. The only required argument is `--config-file`, which should point to your customized YAML configuration file. For example:

```bash
fastreid-train --config-file fastreid/configs/Market1501/bagtricks_R50.yml
```

During the training process, progress will be logged to the console. All generated artifacts, including model checkpoints (`.pth` files) and log files, will be saved to the directory specified by the `OUTPUT_DIR` variable in your configuration file.

### Step 4: Model Evaluation

Once a model has been trained, its performance can be quantitatively assessed using the `fastreid-test` command. This requires the same config file and an `--eval-only` flag. The path to the trained model weights must also be specified by overriding the `MODEL.WEIGHTS` parameter.

```bash
fastreid-test --config-file fastreid/configs/Market1501/bagtricks_R50.yml --eval-only MODEL.WEIGHTS /path/to/your/trained/model.pth
```

The script will then output standard ReID evaluation metrics, such as mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) ranks.

### Step 5: Inference with Custom Images

To extract deep feature vectors from a custom set of images, the `fastreid-demo` tool is provided. This is ideal for building a gallery for similarity search or for deploying a ReID system.

```bash
fastreid-demo --config-file /path/to/config.yml --input "/path/to/images/*.jpg" --output /path/to/output_dir MODEL.WEIGHTS /path/to/model.pth
```

  - `--input`: A glob pattern that matches all your input images.
  - `--output`: The directory where extracted feature vectors will be saved.

-----

## Model Zoo Compatibility

A significant challenge when working with legacy projects is the incompatibility of pre-trained models with updated codebases. `newFastReID` directly solves this problem with the `generate_config.py` script.

**Problem**: A `.pth` model file from the original FastReID model zoo is available, but the exact YAML configuration file required to load it is missing.

**Solution**: The provided script can be used to automatically generate a compatible configuration.

**Usage**:
Execute the script from your terminal. It requires a `--model-path` argument pointing to the pre-trained model file and a `--save-path` argument for the output YAML file.

```bash
python generate_config.py --model-path /path/to/model.pth --save-path /path/to/generated_config.yml
```

The script performs an introspection of the model's architecture to deduce the original configuration settings and serializes this information into a well-formed YAML file, ready for immediate use.

-----

## Information for Developers

For those looking to contribute to or extend the library's functionality, `newFastReID` offers a straightforward, registry-based system.

  - **Project Structure**: The codebase is organized logically to separate concerns. Core logic is contained within the `fastreid/` directory, which includes subdirectories for configuration, data handling, modeling, and more.
  - **Extensibility**: The library is designed to be modular. New components, such as datasets or model backbones, can be added by creating a new file in the appropriate directory and registering the new component with the corresponding registry decorator (e.g., `@DATASET_REGISTRY.register()`). This makes the new component available for use in configuration files without needing to modify the core training or evaluation code.

-----

## License

This project is distributed under the Apache License 2.0. Please see the `LICENSE` file for full details.

## Acknowledgements

This work is a fork of and builds upon the significant contributions of the original [FastReID](https://github.com/JDAI-CV/fast-reid) library. We are grateful to the original authors for providing a strong foundation for the ReID research community.

## Citation

If `newFastReID` is used in academic work, we kindly request that the original FastReID paper be cited:

```bibtex
@article{he2020fastreid,
  title={FastReID: A Pytorch Toolbox for General Instance Re-identification},
  author={He, Lingxiao and Liao, Xingyu and Liu, Wu and Liu, Xinchen and Cheng, Peng and Mei, Tao},
  journal={arXiv preprint arXiv:2006.02631},
  year={2020}
}
```
