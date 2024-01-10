# dynamic-fusion

# Installation

## Requirements

- Python 3.9.*

## Installation
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

# Troubleshooting

#### UI issues
If you are using conda and the UI is looking weird, try:
```
conda remove --force tk
```
See [this issue](https://github.com/ContinuumIO/anaconda-issues/issues/6833#issuecomment-974266793).

