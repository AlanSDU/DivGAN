
 # DivGAN


# architecture of DualGAN

# How to setup

## Prerequisites

* Python (2.7 or later)

* numpy

* scipy

* NVIDIA GPU + CUDA 8.0 + CuDNN v5.1

* TensorFlow 1.0 or later

* unzip


# Getting Started
## steps

* clone this repo:

```
git clone https://github.com/AlanSDU/DivGAN.git

cd DivGAN
```

* download datasets (e.g., sketch-photo), run:

```
bash ./datasets/download_dataset.sh sketch-photo
```

```
python main.py --phase train --dataset_name sketch-photo --image_size_l 256 --image_size_l 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --lambda_Sim_loss 20.0 --A_channels 3 --B_channels 3 --suffix jpg --use_resnet '' --ngf 64
```

* test the model:

```
python main.py --phase test --dataset_name sketch-photo --image_size_l 256 --image_size_l 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --lambda_Sim_loss 20.0 --A_channels 3 --B_channels 3 --suffix jpg --use_resnet '' --ngf 64
```


# datasets
some datasets can also be downloaded manually from the website. Please cite their papers if you use the data. 

facades: http://cmp.felk.cvut.cz/~tylecr1/facade/

sketch: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/

maps: https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c

oil-chinese:  http://www.cs.mun.ca/~yz7241/, jump to http://www.cs.mun.ca/~yz7241/dataset/

day-night: http://www.cs.mun.ca/~yz7241/dataset/

monet2photo: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/


