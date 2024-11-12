# MultiviewFER4Child


> [**Multi-view Facial Expressions Analysis of Autistic Children in Social Play**]<br>
> Jiabei Zeng *, Yujian Yuan *, Lu Qu, Fei Chang, Xuran Sun, Jinqiuyu Gong, Xuling Han, Min Liu, Hang Zhao, Qiaoyun Liu, Shiguang Shan, Xilin Chen <br> *: equal contribution.  <br>Institute of Computing Technology, Chinese Academy of Sciences;
 University of Chinese Academy of Sciences; East China Normal University


## 📰 News
**[2024.11.11]** Codes are released now. We are working on optimizing the codes. <br>
**[2024.10.9]** Code and dataset feature will be released here. Welcome to **watch** this repository for the latest updates.


## ➡️ Dataset feature
Considering the privacy of the participants in the dataset, we cannot release the original video datasets. 
To help the future work on this task, we release the extracted emotion feature of the kids of each frame.
| View                         |                                                    Link                                                    |
|:------------------------------------|:-------------------------------------------------------------------------------------------------------:| 
| Multiview*    					   |     [OneDrive](https://1drv.ms/f/s!Atl7YPj4ORSRjfAdcQ10K01tQ6pRYQ?e=PNNn2i)|
| kangroo                    |     [OneDrive](https://1drv.ms/f/s!Atl7YPj4ORSRjfAcWSJoqNttXcHaQg?e=mraNW2)   |
| insect          |     [OneDrive](https://1drv.ms/u/s!Atl7YPj4ORSRjfAvi5PzyOXDMKqHsw?e=6y8lL5)    | 
| bird          |     [OneDrive](https://1drv.ms/u/s!Atl7YPj4ORSRjfAvi5PzyOXDMKqHsw?e=6y8lL5)    | 
| koala          |     [OneDrive](https://1drv.ms/u/s!Atl7YPj4ORSRjfAvi5PzyOXDMKqHsw?e=6y8lL5)    | 

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

## 🚀 Getting started






## 🤝 Acknowledgement
This work is supported by National Natural Science Foundation of China (No. 62176248), Science Foundation of the Shanghai Education Commission: Major Program(2023SKZD07), China Postdoctoral Science Foundation(2023M731104).
