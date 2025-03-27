# MultiviewFER4Child

> [**Multi-view Facial Expressions Analysis of Autistic Children in Social Play**]<br>
> Jiabei Zeng *, Yujian Yuan *, Lu Qu, Fei Chang, Xuran Sun, Jinqiuyu Gong, Xuling Han, Min Liu, Hang Zhao, Qiaoyun Liu, Shiguang Shan, Xilin Chen <br> *: equal contribution.  <br>Institute of Computing Technology, Chinese Academy of Sciences;
 University of Chinese Academy of Sciences; East China Normal University


## 📰 News
**[2025.3.27]** Codes are released now. We are working on optimizing the rest codes. <br>
**[2025.3.12]** Dataset features are available now.  <br>
**[2024.10.9]** Code and dataset features will be released here. Welcome to **watch** this repository for the latest updates.


## ➡️ Dataset feature
Considering the privacy of the participants in the dataset, we cannot release the original video datasets. 
To help the future work on this task, we release the extracted emotion feature of the kids of each frame.
| View                         |                                                    Link                                                    |
|:------------------------------------|:-------------------------------------------------------------------------------------------------------:| 
| Multiview*    					   |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/EQwg8N-zKMpGvv2vAwSgaXsBo5keMxojx4euxmVLNotzfA)|
| View1                    |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/ERgnDaWxMddKoVUtuU2HKvsB7tYCPmqaG-QlByt3E0G5tw?e=rpNl6h)   |
| View2          |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/Ed8Z_7xxN9tBg9N5ot7mlXYB5h44yv8ihYZsj7039z4GkA?e=0IsO5e)    | 
| View3          |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/Ee88rRjt081Dl2UZF_LEbrQB94WaSlC5D_Xp7BE5O6qOQQ?e=fl2Va5)    | 
| View4          |     [OneDrive](https://1drv.ms/u/c/911439f8f8607bd9/EVMK3IfnnIJCuqrhZBglmLwBSores9tKrAZpdqungx2DWQ?e=qC3Zec)    | 

*: the multiview facial expression is predicted by choosing the most frontal face among the four views.

## 🔨 Installation

1. (Optional) Creating conda environment

```bash
conda create -n mtfer python=3.8.12
conda activate mtfer
```

2. Download the packages in requirements.txt 

```bash
pip install -r requirements.txt 
```

3. Download this repo. 
```bash
git clone https://github.com/Yujianyuan/MultiviewFER4Child.git
cd MultiviewFER4Child
```

## 🤝 Acknowledgement
This work is supported by National Natural Science Foundation of China (No. 62176248), Science Foundation of the Shanghai Education Commission: Major Program(2023SKZD07), China Postdoctoral Science Foundation(2023M731104).
