# STIM

This repository is the official implementation of "[Address Anomalies at Critical Crossroads for Graph Anomaly Detection](https://ieeexplore.ieee.org/abstract/document/11183627)", accepted by TKDE.

![](https://github.com/Junyi-Yan/Junyi-Yan.github.io/blob/main/Picture/TKDE2025.png)

# Overview
Our implementation for STIM is based on PyTorch. 


# Requirments
This code requires the following:

- Python==3.8
- PyTorch==1.9.0+cu111
- Pytorch Geometric==2.3.0
- Numpy==1.21.2
- Scipy==1.9.3
- Scikit-learn==1.1.2
- NetworkX==2.8.8
- OGB==1.3.5
- DGL==0.4.3 
- DGL-cu111==0.6.1 (Do not use the version which is newer than that!)

# Usage
Step1: Pre-processing
```
python preprocessing.py
```
Step2: Anomaly Detection
```
python run.py
```

# Baselines
All baselines and their URLs are as follows:  
- ANOMALOUS [Paper](https://www.ijcai.org/proceedings/2018/488)[Code](https://github.com/zpeng27/ANOMALOUS)


# Cite
If you compare with, build on, or use aspects of this work, please cite the following:

```
@article{yan2025address,  
  title={Address Anomalies at Critical Crossroads for Graph Anomaly Detection},  
  author={Yan, Junyi and Zuo, Enguang and Liang, Ke and Liu, Meng and Li, Miaomiao and Liu, Xinwang and Lv, Xiaoyi and Lu, Kai},  
  journal={IEEE Transactions on Knowledge and Data Engineering},  
  year={2025},  
  publisher={IEEE}  
}
```
