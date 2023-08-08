# Knowledge Distillation for Anomaly Detection

This repository contains the code for generating the results of *Knowledge Distillation for Anomaly Detection* paper, presented at the 31th European Symposium on Artificial Neural Networks (ESANN 2023) by _Adrian Alan Pol\*, Ekaterina Govorkova\*, Sonja Gronroos\*, Nadezda Chernyavskaya, Philip Harris, Maurizio Pierini, Isobel Ojalvo and Peter Elmer_. The preprint is available [HERE](https://arxiv.org/).

## Usage Instructions
Setup the environment, and install the requierments.
```
conda create -n "kd4ad" python=3.9.2
conda activate kd4ad
pip install -r requirements.txt
```
Also, install `torch==2.0.0` and `tensorflow==2.6.0` with the CUDA enabled (when running on GPUs).

### Experiments
You can run experiments as follows (example for the baseline):
```
mkdir results-baseline
python3 experiment-baseline.py
```
