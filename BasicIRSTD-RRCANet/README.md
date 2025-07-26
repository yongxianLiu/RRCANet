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
* **Run **`train.py`** to perform network training. Example for training [model_name] on [dataset_name] datasets:**
  ```
  python train.py --model_names RRCANet --dataset_names DenseSIRST
  ```
* **Checkpoints and Logs will be saved to **`./log/`**:**
  ```
  ├──./log/
  │    ├── [dataset_name]
  │    │    ├── [model_name]_xxxxx.pth.tar
  ```
<be>

## Test
* **Run **`test.py`** to perform network inference and evaluation. Example for test [model_name] on [dataset_name] datasets:**
  ```
  python test.py --model_names RRCANet --dataset_names DenseSIRST --save_img True
  ```

* **Predicted pictures will be saved to **`./results/`**, Pd and Fa will be saved to **`./test_ROC/`**:**
  ```
  ├──./results/
  │    ├── [dataset_name]
  │    │    ├── [model_name]
  |    |    |    |—— xxxxx.png
  |    |    |    |—— xxxxx.png
  ```
<be>

## Abstract
Infrared small target detection is a challenging task due to its unique characteristics (e.g., small, dim, shapeless and changeable). Recently published CNN-based methods have achieved promising performance with heavy feature extraction and fusion modules. To achieve efficient and effective detection, we propose a recurrent reusable-convolution attention network (RRCA-Net) for infrared small target detection. Specifically, RRCA-Net incorporates reusable-convolution block (RuCB) in a recurrent manner without introducing extra parameters. With the help of the repetitive iteration in RuCB, the high-level information of small targets in the deep layers can be well maintained and further refined. Then, a dual interactive attention aggregation module (DIAAM) is proposed to promote the mutual enhancement and fusion of refined information. In this way, RRCA-Net can both achieve high-level feature refinement and enhance the correlation of contextual information between adjacent layers. Moreover, to achieve steady convergence, we design a target characteristic inspired loss function (DpT-k loss) by integrating physical and mathematical constraints. Experimental results on three benchmark datasets (e.g. NUAA-SIRST, IRSTD-1k, DenseSIRST) demonstrate that our RRCA-Net can achieve comparable performance to the state-of-the-art methods while maintaining a small number of parameters, and act as a plug and play module to introduce consistent performance improvement for several popular IRSTD methods.

## Model
![Image text](https://github.com/yongxianLiu/RRCANet/blob/main/RRCA-Net/Fig/network.png)


## Results

#### Quantitative Results
| Model             | Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6))|
| -------------     |:-------------:|:-----:|:-----:| :-----:|
|                   | IRSTD-1K      | 68.73  | 92.52 | 6.68 |
| RRCA-Net-ResNet18 | NUAA-SIRST    | 75.41  | 98.10 | 7.56 |
|                   | Dense-SIRST   | 82.15  | 98.23 | 1.81 |
| [[Weights]](https://pan.baidu.com/s/1ie0GGTPoGVMex8n6b0lqkw?pwd=qz5p)|

#### Transfer Results
![Image text](https://github.com/yongxianLiu/RRCANet/blob/main/RRCA-Net/Fig/Transfer_results.png)

#### Qualitative Results
![Image text](https://github.com/yongxianLiu/RRCANet/blob/main/RRCA-Net/Fig/visual2d.png)



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
