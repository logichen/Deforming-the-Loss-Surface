# Deforming-the-Loss-Surface
Source code of the paper ``Deforming the Loss Surface''
## Requirements

- python **>=3.7**
- pytorch **>=1.1.0**
- easydict **>=1.9**
- future **>=0.17.1**
- tensorboard **>=1.4.0**
- numpy **>=1.18**
- cxxfilt **>=0.2.0**
- tqdm **>=4.28.1**
- pyyaml **>=5.1**
- pytest **>=3.5.1**

```bash
pip install -r requirements.txt
```
## Usage 
### Preparation for Datasets
For CIFAR models, download the CIFAR-10 and CIFAR-100 datasets and put them into the `\data` folder. For ImageNet models, change the path of the dataset by modifying `dataset_dir` in `\configs\imagenet\model_name.yaml`. 

### CIFAR
#### PreResNets, DensNets, and SE-ResNeXt-29 (16 x 64 d):
```bash
python -u train.py --work-path ./experiments/cifar10/preresnet20
```
```bash
cd ./PreResNets_DenseNets_ResNext
python -u train.py --work-path ./experiments/cifar10/preresnet20
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
