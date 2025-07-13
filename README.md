<div align="center">

# HORT: Monocular Hand-held Objects Reconstruction with Transformers

[Zerui Chen](https://zerchen.github.io/)<sup>1</sup> &emsp; [Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>2</sup> &emsp; [Shizhe Chen](https://cshizhe.github.io/)<sup>1</sup> &emsp; [Cordelia Schmid](https://cordeliaschmid.github.io/)<sup>1</sup>

<sup>1</sup>WILLOW, INRIA Paris, France <br>
<sup>2</sup>Imperial College London, UK

<a href='https://zerchen.github.io/projects/hort.html'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2503.21313'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/spaces/zerchen/HORT'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>
</div>

This is the training code of **[HORT](https://zerchen.github.io/projects/hort.html)**, an state-of-the-art hand-held object reconstruction algorithm.

## Installation 👷
```
git clone https://github.com/zerchen/hort_train.git
cd hort_train
```

It is suggested to use an anaconda encironment to install the the required dependencies:
```bash
conda create --name hort python=3.12
conda activate hort

conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# Install requirements
pip install -r requirements.txt
conda install pytorch3d-0.7.8-py312_cu121_pyt241.tar.bz2 # https://anaconda.org/pytorch3d/pytorch3d/files?page=2
cd common/networks/tgs/models/snowflake/pointnet2_ops_lib && python setup.py install
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/mano/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).

## Data preparation 📈
Please organize the data into structures like this (take the ObMan dataset as an example):
```
   ${ROOT}/datasets/obman
   └── splits
       obman.py
       data
        ├── val
        ├── train
        └── test
            ├── rgb
            ├── mesh_hand
            ├── mesh_obj
```
Then, execute corresponding preprocessing code to generate training files:
```bash
python preprocess/cocoify_obman.py
```

## Training 💻
Then, launch the training script as follows.
```bash
cd tools
# training, and testing will be launched automatically when training finishes
bash dist_train.sh 4 1234 -e ../playground/object_pc_dino/experiments/obman_141k.yaml --gpu 0-3
# evaluation
python eval.py -e ${OUTPUT_DIR}
```

## Acknowledgements
Parts of the code are based on [WiLoR](https://github.com/rolpotamias/WiLoR), [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet) and [gSDF](https://github.com/zerchen/gSDF).

## License 📚
HORT is licensed under MIT License. This repository also depends on [WiLoR](https://github.com/rolpotamias/WiLoR), [Ultralytics library](https://github.com/ultralytics/ultralytics) and [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses.
## Citation  📝
If you find HORT useful for your research, please consider citing our paper:

```bibtex
@InProceedings{chen2025hort,
  title={{HORT}: Monocular Hand-held Objects Reconstruction with Transformers},
  author={Chen, Zerui and Potamias, Rolandos Alexandros and Chen, Shizhe and Schmid, Cordelia},
  booktitle={ICCV},
  year={2025}
}
```
