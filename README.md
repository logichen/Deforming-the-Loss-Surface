# Deforming-the-Loss-Surface
Source code of the paper ``Deforming the Loss Surface''
## Requirements and Usage 
### Requirements

- python **>=3.7**
- pytorch **>=1.1.0**
- easydict **>=1.9**
- future **>=0.17.1**
- tensorboard **>=1.4.0**
- pyyaml


```bash
pip install -r requirements.txt
```
### Usage 

#### PreResNets
simply run the cmd for the training:

```bash
cd 
```

```bash
## 1 GPU for lenet
CUDA_VISIBLE_DEVICES=0 python -u train.py --work-path ./experiments/cifar10/lenet

## resume from ckpt
CUDA_VISIBLE_DEVICES=0 python -u train.py --work-path ./experiments/cifar10/lenet --resume

## 2 GPUs for resnet1202
CUDA_VISIBLE_DEVICES=0,1 python -u train.py --work-path ./experiments/cifar10/preresnet1202

## 4 GPUs for densenet190bc
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py --work-path ./experiments/cifar10/densenet190bc
``` 