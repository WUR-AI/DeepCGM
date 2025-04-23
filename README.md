# DeepCGM: A Knowledge-Guided Deep Learning Crop Growth Model

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

Crop growth modeling is essential for understanding and predicting agricultural outcomes. Traditional **process-based crop models**, like ORYZA2000, are effective but often suffer from oversimplification and parameter estimation challenges. **Machine learning methods**, though promising, are often criticized for being "black-box" models and requiring large datasets that are frequently unavailable in real-world agricultural settings.

**DeepCGM** addresses these limitations by integrating knowledge-guided constraints into a deep learning model to ensure physically plausible crop growth simulations, even with sparse data.

This repository contains the code and resources for the paper: [Knowledge-guided machine learning with multivariate sparse data for crop growth modelling](https://www.sciencedirect.com/science/article/pii/S0378429025001777)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Model Architecture](#model-architecture)
- [Data](#data)
- [Train the Model](#train-the-model)
- [Training Flowchart](#training-flowchart)
- [Evaluate the Model](#evaluate-the-model)
- [License](#-license)

## Features

- **Mass-Conserving Deep Learning Architecture**: Adheres to crop growth principles such as mass conservation to ensure physically realistic predictions.
- **Knowledge-Guided Constraints**: Includes crop physiology and model convergence constraints, enabling accurate predictions with sparse data.
- **Improved Accuracy**: Outperforms traditional process-based models and classical deep learning models on real-world crop datasets.
- **Multivariable Prediction**: Simulates multiple crop growth variables (e.g., biomass, leaf area) in a single framework.

## Installation

To install the dependencies, clone the repository and install the required packages using the command below:

**conda**

```bash
git clone https://github.com/yourusername/DeepCGM.git
cd DeepCGM
conda create -n DeepCGM
conda activate DeepCGM
pip install -r .\requirements.txt
```

## Repository Structure

- **`requirements.txt`**: Requirements.
- **`train.py`**: Script to train the different model.
- **`utils.py`**: Utility functions for data preprocessing and model support.
- **`fig_5.py`, `fig_6.py`, `fig_7.py`, etc.**: Scripts to generate figures for model results.
- **`models_aux`**: Folder containing models.
  - **`DeepCGM.py`** is the DeepCGM model 100% following the [detail process of DeepCGM](figure/DeepCGM_detail.svg)
  - **`DeepCGM_fast.py`** improve the model speed by combining the gate calculation according to [this suggestion](https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/) and by combining the redistribution calculation.
  - **`MCLSTM.py`** and **`MCLSTM_fast.py`** are raw MCLSTM and speed improved MCLSTM
- **`format_dataset`**: Formatted dataset.
- **`figure`**: Folder for storing figures generated during model evaluation and analysis.

## Model Architecture

DeepCGM is a **deep learning-based crop growth model** with a mass-conserving architecture (detail refers to [detail process of DeepCGM](figure/DeepCGM_detail.svg)). The architecture ensures that simulated crop growth adheres to physical principles, including:

![Model Structure](figure/DeepCGM.svg)

## Data

DeepCGM operates on **time series data** representing crop growth cycles.

* **Input Data:** Meteorological variables (e.g., daily solar radiation, maximum temperature, minimum temperature), management information (e.g., cumulative nitrogen applied), and optionally, simulated variables like crop development stage (DVS) from a model like ORYZA2000.
* **Output (Target) Data:** Measured crop variables used for training and evaluation, such as Plant Area Index (PAI), biomass of individual organs (leaf, stem, grain), total above-ground biomass (WAGT), and final yield.

## Train the Model

Run the `train.py` script to train the model using your formatted data:

```bash
python train.py --model DeepCGM --target spa --input_mask 1 --convergence_loss 1 --tra_year 2018
```

You can modify the training parameters, such as model type, knowledge triggers, and training years as following arguments:

- **--model**: Specifies the model type (`NaiveLSTM`,`MCLSTM`, `DeepCGM`).
- **--target**: Specifies the training label ( `spa` for sparse dataset and `int` for interpolated dataset).
- **--input_mask**: Enables the input mask (`1` to enable, `0` to disable).
- **--convergence_trigger**: Enables the convergence_loss (`1` to enable, `0` to disable).
- **--tra_year**: Specifies the training year (e.g., `2018` and `2019`).

## Training flowchat

The `fitting loss`, `convergence loss` and `input mask` can be used in training DeepCGM

![Training flowchart](figure/Training.svg)

## Evaluate the Model

Use the figure scripts (e.g., `fig_5.py`, `fig_12.py`, etc.) to generate visualizations of the model's performance. Example:

```bash
python fig_5.py
```

**Crop growth simulation results of models trained by different data and strategies**
![Time series result](figure/Fig.5%20Crop%20growth%20simulation%20results.svg)

```bash
python fig_12.py
```

**DeepCGM outperforms traditional process-based models (Normlized Index):**

![Radar chart](figure/Fig.12.%20The%20normalized%20index%20of%20different%20models%20trained%20by%20different%20strategies%20on%20sparse%20dataset.svg)

More results are saved in the `figure` folder, and detailed evaluation figures are generated using the provided scripts.

## 📄 License

This project is licensed under the **CC BY-NC 4.0 License** — for **non-commercial research and academic use only**.

See [LICENSE](./LICENSE.md) for full details.

For commercial use, please contact: [your-email@example.com]
