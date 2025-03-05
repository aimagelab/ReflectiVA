
<div align="center">
  <h1>Reflective LLaVA (ReflectiVA)</h1>
  <h2>Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering

  (CVPR 2025)
  </h2>
   
</div>

<br></br>
<p align="center">
  <img src="images/model.png" alt="reflectiva" width="820" />

</p> 

This repository contains the reference code for the paper [Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering](https://arxiv.org/abs/2411.16863).

## Table of Contents

1. [Citation](#citation)
2. [Overview](#overview)
3. [Installation](#installation)
4. [Model](#model)
5. [Knowledge Based](#knowledge-based)
6. [Inference](#inference)

## Citation

Please cite with the following BibTeX:
```
@inproceedings{cocchi2024augmenting,
  title={{Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering}},
  author={Cocchi, Federico and Moratelli, Nicholas and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Overview
Multimodal LLMs (MLLMs) are the natural extension of
large language models to handle multimodal inputs, combining text and image data. 
They have recently garnered attention due to their capability to address complex tasks involving both modalities. 
However, their effectiveness is limited to the knowledge acquired during training, which restricts their practical utility. 
In this work, we introduce a novel method to enhance the adaptability of MLLMs by integrating external knowledge sources. 
Our proposed model, Reflective LLaVA (```ReflectiVA```), utilizes reflective tokens to dynamically determine the need for external knowledge 
and predict the relevance of information retrieved from an external database, ultimately enables the MLLM to manage external knowledge 
while preserving fluency and performance on tasks where external knowledge is not needed.

## Installation
To create the conda environment named reflectiva use the following instructions.
With this environment you have all the packages to run the code inside this repo. 
```
conda create -n reflectiva python==3.8.16
conda activate reflectiva
pip install -r requirements.txt
```

## Model

You can access the official model weights for the [ReflectiVA model](https://huggingface.co/aimagelab/ReflectiVA) on 🤗 Hugging Face.

## Knowledge Based
Coming soon ...

## Inference
Coming soon ...
