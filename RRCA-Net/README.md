# Recurrent Reusable-Convolution Attention Network for Infrared Small Target Detection

Pytorch implementation of our "RRCANet: Recurrent Reusable-Convolution Attention Network for Infrared Small Target Detection". [[paper]](https://arxiv.org/pdf/2506.02393)


## Requirements
- Python 3.9
- pytorch 1.10.1, torchvision 0.11.2, torchaudio 0.10.1, Cuda 11.1 or higher
<br><br>

## Datasets
* SIRST &nbsp; [[download dir]](https://github.com/YimianDai/sirst) &nbsp; [[paper]](https://arxiv.org/pdf/2009.14530.pdf)
* IRSTD-1K &nbsp; [[download dir]](https://github.com/RuiZhang97/ISNet) &nbsp; [[paper]](https://ieeexplore.ieee.org/document/9880295)
* DenseSIRST &nbsp; [[download dir]](https://github.com/GrokCV/DenseSIRST) &nbsp; [[paper]](https://arxiv.org/abs/2407.20078)

* **The organization of our dataset is as follows:**
  ```
  ├──./dataset/
  │    ├── IRSTD-1K
  │    │    ├── 80_20
  │    │    │    ├── train.txt
  │    │    │    ├── test.txt
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── value_result
  │    │    ├── visulization_result
  │    ├── NUAA-SIRST
  │    │    ├── 50_50
  │    │    │    ├── train.txt
  │    │    │    ├── test.txt
  │    │    ├── images
  │    │    │    ├── Misc_1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── Misc_1.png
  │    │    │    ├── ...
  │    │    ├── value_result
  │    │    ├── visulization_result
  │    ├── ...  
  ```

## Train
* **Run **`train.py`** to perform network training. The training settings are in the **`model/parse_args_train.py`**.**
  ```
  python train.py
  ```
* **Checkpoints and Logs will be saved to **`./result/`**:**
  ```
  ├──./result/
  │    ├── [dataset_name]_[model_name]_[time]
  │    │    ├── [model_name]_[dataset_name]_epoch.pth.tar
  │    │    ├── [model_name]_[dataset_name]IoU.log
  │    │    ├── [model_name]_[dataset_name]_other_metric.log
  │    │    ├── train_log.txt
  ```
<be>

## Test
* **Run **`test.py`** to perform network testing. The testing settings are in the **`model/parse_args_test.py`**.**
  ```
  python test.py
  ```
<br>

## Citiation
```
@article{liu2025rrcanet,
  title={RRCANet: Recurrent Reusable-Convolution Attention Network for Infrared Small Target Detection},
  author={Liu, Yongxian and Li, Boyang and Liu, Ting and Lin, Zaiping and An, Wei},
  journal={arXiv preprint arXiv:2506.02393},
  year={2025}
}
```
<br>


## Acknowledgement
**Thanks for [DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection) and [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD).**
<br><br>

## Contact
**Welcome to raise issues or email to yongxian23@nudt.edu.cn for any question.**
