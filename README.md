# physics-informed-metasurface

## Environment Variables
Create `.env` file in the same level directory as `main.py`

and set below
```shell
DATA_DIR={/to/data/}
PRETRAINED_MODEL_PATH={/to/pretrained_model/}
```
e.g.
```shell
DATA_DIR=/data_dir
PRETRAINED_MODEL_PATH=/data_dir/pretrained.pt
```

## Requirements

```shell
pip install -r requirements.txt
```
### DeflectorGym

```shell
git clone https://github.com/kc-ml2/deflector-gym.git
cd deflector-gym
pip install -e .
```

## Run
```shell
git clone https://github.com/jLabKAIST/physics-informed-metasurface.git
cd physics-informed-metasurface
python main.py
```
