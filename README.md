<div align="center">
    <h2>
        Change-Agent: Towards Interactive Comprehensive Change Interpretation and Analysis from Change Detection and Change Captioning
    </h2>
</div>
<br>
<div align="center">
  <img src="resource/Change_Agent.png" width="800"/>
</div>
<br>
<div align="center">
  <a href="https://arxiv.org/abs/2403.19654">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
</div>

[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2403.19654-b31b1b.svg)](https://arxiv.org/abs/2403.19646)



## Welcome to our repository! 

This repository contains the PyTorch implementation of our PromptCC model in the paper: "A Decoupling Paradigm with Prompt Learning for Remote Sensing Image Change Captioning". 

For more information, please see our published paper in [[IEEE](https://ieeexplore.ieee.org/document/10271701)]  ***(Accepted by TGRS 2023)***

## Overview
- Considering the specificity of the RSICC task, PromptCC employs a novel decoupling paradigm and deeply integrates prompt learning and pre-trained large language models.
- This repository will encompass all aspects of our code, including **training, inference, computation of evaluation metrics, as well as the tokenization and word mapping** used in our work.

<div align="center">
<img src="./Example/Prompt_CC.png" width="600"></img>
</div>

[//]: # (## Contributions)

[//]: # (- **Decoupling Paradigm**: The previous methods predominantly adhere to the encoder-decoder framework directly borrowed from the image captioning field, overlooking the specificity of the RSICC task. Unlike that, we propose a decoupling paradigm to decouple the RSICC task into two issues: whether and what changes have occurred. Specifically, we propose a pure Transformer-based model in which an image-level classifier and a feature-level encoder are employed to address the above two issues. The experiments validate the effectiveness of our approach. Furthermore, in Section IV-G, we discuss the advantages of our decoupling paradigm to demonstrate that the new paradigm has a broad prospect and is more proper than the previous coupled paradigm for the RSICC task.)

[//]: # (- **Integration of prompt learning and pre-trained large language models**: To our knowledge, we are the **first** to introduce prompt learning and the LLM into the RSICC task. To fully exploit their potential in the RSICC task, we propose a multi-prompt learning strategy which can effectively exploit the powerful abilities of the pre-trained LLM, and prompt the LLM to know whether changes exist and generate captions. Unlike the previous methods, our method can generate plausible captions without retraining a language decoder from scratch as the caption generator. Lastly, with the recent emergence of various LLMs, we believe that LLMs will attract broader attention in the remote sensing community in the forthcoming years. We aspire for our paper to inspire future advancements in remote sensing research.)

[//]: # (- **Experiments**: Experiments show that our decoupling paradigm and the multi-prompt learning strategy are effective and our model achieves SOTA performance with a significant improvement. Besides, an additional experiment demonstrates our decoupling paradigm is more proper than the previous coupled paradigm for the RSICC task.)
