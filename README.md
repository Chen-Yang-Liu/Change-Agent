<div align="center">
    
<h1><a href="https://ieeexplore.ieee.org/document/10591792">Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis</a></h1>

**[Chenyang Liu](https://chen-yang-liu.github.io/), [Keyan Chen](https://kyanchen.github.io), [Haotian Zhang](https://scholar.google.com/citations?user=c7uR6NUAAAAJ), [Zipeng Qi](https://scholar.google.com/citations?user=KhMtmBsAAAAJ), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), and [Zhenwei Shi*✉](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**

</div>

## Share us a :star: if you're interested in this repo

This repository contains the official PyTorch implementation of the paper: "**Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis**" in [[IEEE](https://ieeexplore.ieee.org/document/10591792)]  ***(Accepted by IEEE TGRS 2024)***

<div align="center">
  <img src="resource/Change_Agent.png" width="400"/>
</div>

## LEVIR-MCI dataset 
- Download the LEVIR_MCI dataset: [LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main) (**Available Now!**).
- The dataset contains bi-temporal images as well as diverse change detection masks and descriptive sentences. It provides a crucial data foundation for exploring multi-task learning for change detection and change captioning.
    <br>
    <div align="center">
      <img src="resource/dataset.png" width="600"/>
    </div>
    <br>
## Training of the multi-level change interpretation model
- The training code of the MCI model will be available **before July 31st**.
    <div align="center">
      <img src="resource/MCI_model.png" width="600"/>
    </div>

### Preparation
- Cd into the ``Multi_change`` folder:
```
cd ./Multi_change
```
- Install the required packages: `pip install -r requirements.txt`
- Download the LEVIR-MCI dataset: [LEVIR-MCI](https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/tree/main)
- The data structure of LEVIR-MCI is organized as follows:

```
├─/DATA_PATH_ROOT/Levir-MCI-dataset/
        ├─LevirCCcaptions.json
        ├─images
             ├─train
             │  ├─A
             │  ├─B
             │  ├─label
             ├─val
             │  ├─A
             │  ├─B
             │  ├─label
             ├─test
             │  ├─A
             │  ├─B
             │  ├─label
```
where folder ``A`` contains pre-phase images, folder ``B`` contains post-phase images, and folder ``label`` contains the change detection masks.

- Extract text files for the change descriptions of each image pair in LEVIR-MCI:

```
python preprocess_data.py
```

## Construction of Change-Agent
- The code will be available
    <br>
    <div align="center">
      <img src="resource/overview_agent.png" width="500"/>
    </div>


## Citation & Acknowledgments
If you find this paper useful in your research, please consider citing:
```
@ARTICLE{Liu_Change_Agent,
  author={Liu, Chenyang and Chen, Keyan and Zhang, Haotian and Qi, Zipeng and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Remote sensing;Feature extraction;Semantics;Transformers;Roads;Earth;Task analysis;Interactive Change-Agent;change captioning;change detection;multi-task learning;large language model},
  doi={10.1109/TGRS.2024.3425815}}

```
