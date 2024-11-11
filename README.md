<div align="center">

# ChatTime: A Multimodal Time Series Foundation Model

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=0000.00000&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/0000.00000)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E)](https://huggingface.co/collections/ChengsenWang/chattime-datasets-6731b504efecc8a6e439741c)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/ChengsenWang/chattime-models-6731b650cb98bc7842713fde)

</div>

## ✨ Introduction

Human experts typically integrate numerical and textual multimodal information to analyze time series. However, most traditional deep learning predictors rely solely on unimodal numerical data, using a fixed-length window for training and prediction on a single dataset, and cannot adapt to different scenarios. The powered pre-trained large language model has introduced new opportunities for time series analysis. Yet, existing methods are either inefficient in training, incapable of handling textual information, or lack zero-shot forecasting capability. In this paper, we innovatively model time series as a foreign language and construct ChatTime, a unified framework for time series and text processing. As an out-of-the-box multimodal time series foundation model, ChatTime provides zero-shot forecasting capability and supports bimodal input/output for both time series and text. We design a series of experiments to verify the superior performance of ChatTime across multiple tasks and scenarios, and create four multimodal datasets to address data gaps. The experimental results demonstrate the potential and utility of ChatTime. Specifically, in the unimodal zero-shot forecasting task, ChatTime achieves 99.9% of the previous state-of-the-art using only 4% of the pre-training data. In the multimodal tasks of context-guided forecasting and time series question answering, ChatTime outperforms the previous state-of-the-art by 3.7% and 36.6%, respectively.

<img src="./img/Arch.png" alt="Arch" style="zoom:50%;" />

### Zero-Shot Time Series Forecasting

<img src="./img/ZSTSF.png" alt="ZSTSF" style="zoom:50%;" />

### Context-Guided Time Series Forecasting

<img src="./img/CGTSF.png" alt="CGTSF" style="zoom:50%;" />

### Time Series Question Answering

<img src="./img/TSQA.png" alt="TSQA" style="zoom:50%;" />

## 📈 Usage

## :floppy_disk: Datasets

## 📝 Citation