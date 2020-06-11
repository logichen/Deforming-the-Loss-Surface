# Deforming-the-Loss-Surface
Source code of the paper ``Deforming the Loss Surface''
## Requirements and Usage 
### Requirements

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
### Usage 
#### Preparation for Datasets
For CIFAR models, download the CIFAR-10 and CIFAR-100 datasets and put them into the `\data` folder. For ImageNet models, change the `dataset_dir` by modifying `\configs\imagenet\`
#### PreResNets
Training the deformation version:

```bash
cd .\Experiments_on_CIFAR\CIFAR10\PreResNet_20\Deformed
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
