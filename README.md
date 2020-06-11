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
Assume that the default path is at `Deforming_the_Loss_Surface/`

### CIFAR
#### PreResNets, DensNets, and SE-ResNeXt-29 (16 x 64 d) on CIFAR
Download the CIFAR-10 and CIFAR-100 datasets and put them into the `./CIFAR_all/PreResNets_DenseNets_ResNext/data` folder for preparation. 
To compare the original PreResNet-20 with the deformed version on CIFAR-100 via the following instructions.
Original:
```bash
cd CIFAR_all/PreResNets_DenseNets_ResNext
python -u train_ori.py --work-path ./experiments/cifar100/preresnet20
```
Deformed:
```bash
cd CIFAR_all/PreResNets_DenseNets_ResNext
python -u train_PreResNet20_C100_deformed.py --work-path ./experiments/cifar100/preresnet20
```

To compare the original PreResNet-110 with the deformed version on CIFAR-10 via the following instructions.
Original:
```bash
cd CIFAR_all/PreResNets_DenseNets_ResNext
python -u train_ori.py --work-path ./experiments/cifar10/preresnet110
```
Deformed:
```bash
cd CIFAR_all/PreResNets_DenseNets_ResNext
python -u train_PreResNet110_C10_deformed.py --work-path ./experiments/cifar10/preresnet110
```
#### ResNets on CIFAR
Download the CIFAR-10 and CIFAR-100 and put them into `./CIFAR_all/ResNets/CIFAR10/data` and `./CIFAR_all/ResNets/CIFAR100/data` respectively. 
To compare the original ResNet-20 with the deformed version on CIFAR-10 via the following instructions.
Original:
```bash
cd CIFAR_all/ResNets/CIFAR10
python -u main_ori.py --model resnet20
```
Deformed:
```bash
cd CIFAR_all/ResNets/CIFAR10
python -u main_ResNet20_C10_deformed.py --model resnet20
```

To compare the original ResNet-110 with the deformed version on CIFAR-100 via the following instructions.
##### Original:
```bash
cd CIFAR_all/ResNets/CIFAR100
python -u main_ori.py --model resnet110
```
Deformed:
```bash
cd CIFAR_all/ResNets/CIFAR100
python -u main_ResNet110_C100_deformed.py --model resnet110
```
#### EfficientNets on CIFAR
To compare the original EficientNet-B0 with the deformed version on CIFAR-10 via the following instructions.
```bash
cd CIFAR_all/EfficientNets
python train_C10_ori.py --model efficientnet_b0 --data_dir path_of_cifar10_data
```
```bash
python train_EficientNetB0_C10_deformed.py --model efficientnet_b0 --data_dir path_of_cifar10_data
```
To compare the original EficientNet-B1 with the deformed version on CIFAR-100 via the following instructions.
```bash
python train_C100_ori.py --model efficientnet_b0 --data_dir path_of_cifar100_data
```
```bash
python train_EficientNetB0_C100_deformed.py --model efficientnet_b0 --data_dir path_of_cifar100_data
```
### ImageNet
Change the path of the ImageNet dataset by modifying `dataset_dir` in `./Experiments_on_ImageNet/XXX/Deformed/configs/model_name.yaml`, where 'XXX' can be replaced by the model name such as 'ResNet18' or 'DenseNet121', and 'model_name.yaml' means .yaml files such as 'resnet_18.yaml'.

To compare the original ResNet-18 with the deformed version on ImageNet via the following instructions.
```bash
cd ./ResNet18/Original
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 9595 train.py --config configs/imagenet/resnet_18.yaml train.output_dir experiments/ori_resnet18_2gpus_120ep train.distributed True train.dataloader.pin_memory True > ori_resnet18_2gpus_120ep.txt 2>&1
```
```bash
cd .. && cd Deformed
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 7658 train.py --config configs/imagenet/resnet_18.yaml train.output_dir experiments/deformed_resnet18_2gpus_120ep train.distributed True train.dataloader.pin_memory True > deformed_resnet18_2gpus_120ep.txt 2>&1
```

To compare the original ResNet-34 with the deformed version on ImageNet via the following instructions.
```bash
cd ./ResNet34/Original
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 5632 train.py --config configs/imagenet/resnet_34.yaml train.output_dir experiments/ori_resnet34_2gpus_120ep train.distributed True train.dataloader.pin_memory True > ori_resnet34_2gpus_120ep.txt 2>&1
```
```bash
cd .. && cd Deformed
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node 2 --master_port 9876 train.py --config configs/imagenet/resnet_34.yaml train.output_dir experiments/deformed_resnet34_2gpus_120ep train.distributed True train.dataloader.pin_memory True > deformed_resnet34_2gpus_120ep.txt 2>&1
```
