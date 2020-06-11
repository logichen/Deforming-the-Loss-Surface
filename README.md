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
To compare the original PreResNet-20 with the deformed version on CIFAR-100 via
```bash
python -u train_ori.py --work-path ./experiments/cifar100/preresnet20
```

```bash
python -u train_PreResNet20_C100_deformed.py --work-path ./experiments/cifar100/preresnet20
```

To compare the original PreResNet-110 with the deformed version on CIFAR-10 via
```bash
python -u train_ori.py --work-path ./experiments/cifar10/preresnet110
```

```bash
python -u train_PreResNet110_C10_deformed.py --work-path ./experiments/cifar10/preresnet110
```
#### ResNets
To compare the original ResNet-20 with the deformed version on CIFAR-10 via
```bash
cd CIFAR10
```
```bash
python -u main_ori.py --model resnet20
```
```bash
python -u main_ResNet20_C10_deformed.py --model resnet20
```

To compare the original ResNet-110 with the deformed version on CIFAR-100 via
```bash
cd CIFAR100
```
```bash
python -u main_ori.py --model resnet110
```
```bash
python -u main_ResNet110_C100_deformed.py --model resnet110
```
#### EfficientNets
To compare the original EficientNet-B0 with the deformed version on CIFAR-10 via
```bash
python train_C100_ori.py --model efficientnet_b0 --data_dir path_of_the_data
```
```bash
python train_EficientNetB0_C10_deformed.py --model efficientnet_b0 --data_dir path_of_the_data
```
