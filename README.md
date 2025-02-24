# RRCANet

Pytorch implementation of our


## Requirements
- Python 3
- pytorch (1.2.0), torchvision (0.4.0) or higher
- numpy, PIL
<br><br>

## Datasets
* SIRST &nbsp; [[download dir]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* IRSTD-1K &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
* DenseSIRST &nbsp; [[download dir]](https://github.com/GrokCV/DenseSIRST &nbsp; [[paper]](https://arxiv.org/abs/2407.20078)

## Train
```bash
python train.py --model_names DNANet ALCNet ACM --dataset_names SIRST3 --label_type 'centroid'

python train.py --model_names DNANet ALCNet ACM --dataset_names SIRST3 --label_type 'coarse'

python train_full.py --model_names DNANet ALCNet ACM --dataset_names SIRST3 --label_type 'full'
```
<br>

## Test
```bash
python test.py --model_names DNANet ALCNet ACM --pth_dirs None --dataset_names SIRST NUDT-SIRST IRSTD-1K

python test.py --model_names DNANet ALCNet ACM --pth_dirs SIRST3/DNANet_full.pth.tar SIRST3/DNANet_LESPS_centroid.pth.tar SIRST3/DNANet_LESPS_coarse.pth.tar SIRST3/ALCNet_full.pth.tar SIRST3/ALCNet_LESPS_centroid.pth.tar SIRST3/ALCNet_LESPS_coarse.pth.tar SIRST3/ACM_full.pth.tar SIRST3/ACM_LESPS_centroid.pth.tar SIRST3/ACM_LESPS_coarse.pth.tar --dataset_names SIRST NUDT-SIRST IRSTD-1K
```
<br>

## Citiation
```
@article
```
<br>


## Acknowledgement
**Thanks for [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection) and [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD).**
<br><br>

## Contact
**Welcome to raise issues or email to yongxian23@nudt.edu.cn for any question.**
