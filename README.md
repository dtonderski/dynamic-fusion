# dynamic-fusion
This repository is about Implicit Image Reconstruction and SpatioTemporal Superresolution using EVS.

# Installation

## Requirements

- Python 3.9.*

## Installation
Install ffmpeg
```
sudo apt install ffmpeg
```
Create a virtual environment:
```
python3 -m venv venv
source ./venv/bin/activate
```
Install `torch~=1.12` and `torchvision~=0.13` for your CUDA version, e.g.:
```
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
```

Install requirements:

```
pip install -r requirements.txt
```

Install package:
```
pip install -e .
```

## Data
The training data was generated using a proprietary Sony EVS Simulator. It will be made available for download here. After downloading, extract it to a directory of your choice and adjust the `yml` configuration files. The default 

# Running
All code is run through a single entry mode, the `main.py` file.

### Training
To train a model, 

# Troubleshooting

#### UI issues
If you are using conda and the UI is looking weird, try:
```
conda remove --force tk
```
See [this issue](https://github.com/ContinuumIO/anaconda-issues/issues/6833#issuecomment-974266793).

