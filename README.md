
 # ICCV paper of DualGAN
<a href="https://arxiv.org/abs/1704.02510">DualGAN: unsupervised dual learning for image-to-image translation</a>

please cite the paper, if the codes has been used for your research.

# architecture of DualGAN

![architecture](https://github.com/duxingren14/DualGAN/blob/master/0.png)

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
git clone https://github.com/duxingren14/DualGAN.git

cd DualGAN
```

```
monet2photo
python main.py --phase train --dataset_name monet2photo --image_size_l 256 --image_size_l 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --lambda_Sim_loss 20.0 --A_channels 3 --B_channels 3 --suffix jpg --use_resnet True --ngf 64
```

* test the model:

```
python main.py --phase test --dataset_name monet2photo --image_size_l 256 --image_size_l 256 --epoch 45 --lambda_A 20.0 --lambda_B 20.0 --lambda_Sim_loss 20.0 --A_channels 3 --B_channels 3 --suffix jpg --use_resnet True --ngf 64
```


# datasets
some datasets can also be downloaded manually from the website. Please cite their papers if you use the data. 

facades: http://cmp.felk.cvut.cz/~tylecr1/facade/

sketch: http://mmlab.ie.cuhk.edu.hk/archive/cufsf/

maps: https://mega.nz/#!r8xwCBCD!lNBrY_2QO6pyUJziGj7ikPheUL_yXA8xGXFlM3GPL3c

oil-chinese:  http://www.cs.mun.ca/~yz7241/, jump to http://www.cs.mun.ca/~yz7241/dataset/

day-night: http://www.cs.mun.ca/~yz7241/dataset/


# Experimental results:

![day2night](https://github.com/duxingren14/DualGAN/blob/master/6.PNG)
![da2ni](https://github.com/duxingren14/DualGAN/blob/master/da2ni.png)
![la2ph](https://github.com/duxingren14/DualGAN/blob/master/la2ph.png)
![ph2la](https://github.com/duxingren14/DualGAN/blob/master/ph2la.png)
![sk2ph](https://github.com/duxingren14/DualGAN/blob/master/sk2ph.png)
![ph2sk](https://github.com/duxingren14/DualGAN/blob/master/ph2sk.png)
![ch2oi](https://github.com/duxingren14/DualGAN/blob/master/ch2oi.png)
![oi2ch](https://github.com/duxingren14/DualGAN/blob/master/oi2ch.png)

# Acknowledgments

Codes are built on the top of pix2pix-tensorflow and DCGAN-tensorflow. Thanks for their precedent contributions!
