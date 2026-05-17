<div align="center">

# ChatTime: A Multimodal Time Series Foundation Model

<a href='https://ojs.aaai.org/index.php/AAAI/article/view/33384'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E)](https://huggingface.co/collections/ChengsenWang/chattime-datasets-6731b504efecc8a6e439741c)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/ChengsenWang/chattime-models-6731b650cb98bc7842713fde)

</div>

## ✨ Introduction

Human experts typically integrate numerical and textual multimodal information to analyze time series. However, most traditional deep learning predictors rely solely on unimodal numerical data, using a fixed-length window for training and prediction on a single dataset, and cannot adapt to different scenarios. The powered pre-trained large language model has introduced new opportunities for time series analysis. Yet, existing methods are either inefficient in training, incapable of handling textual information, or lack zero-shot forecasting capability. In this paper, we innovatively model time series as a foreign language and construct ChatTime, a unified framework for time series and text processing. As an out-of-the-box multimodal time series foundation model, ChatTime provides zero-shot forecasting capability and supports bimodal input/output for both time series and text. We design a series of experiments to verify the superior performance of ChatTime across multiple tasks and scenarios, and create four multimodal datasets to address data gaps. The experimental results demonstrate the potential and utility of ChatTime. Specifically, in the unimodal zero-shot forecasting task, ChatTime achieves 99.9% of the previous state-of-the-art using only 4% of the pre-training data. In the multimodal tasks of context-guided forecasting and time series question answering, ChatTime outperforms the previous state-of-the-art by 3.7% and 36.6%, respectively.

![](./img/architecture.png)

## 📈 Usage

We present three minimal examples showing how to perform the multimodal time series analysis using the ChatTime model. The code and corresponding results are available in the [notebook](./demo.ipynb).

### Zero-Shot Time Series Forecasting

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.model import ChatTime

dataset = "Traffic"
hist_len = 120
pred_len = 24
model_path = "ChengsenWang/ChatTime-1-7B-Chat"

df = pd.read_csv(f"./dataset/{dataset}.csv")
hist_data = np.array(df["Hist"].apply(eval).values.tolist())[:, -hist_len:][0]
pred_data = np.array(df["Pred"].apply(eval).values.tolist())[:, :pred_len][0]

model = ChatTime(hist_len=hist_len, pred_len=pred_len, model_path=model_path)

out = model.predict(hist_data)

hist_x = np.linspace(0, hist_len-1, hist_len)
pred_x = np.linspace(hist_len, hist_len+pred_len-1, pred_len)

plt.figure(figsize=(8, 2), dpi=500)
plt.plot(hist_x, hist_data, color='#000000')
plt.plot(pred_x, pred_data, color='#000000', label='true')
plt.plot(pred_x, out, color='#FF7F0E', label='pred')
plt.axvline(hist_len, color='red')
plt.legend(loc="upper left")
plt.show()

```

### Context-Guided Time Series Forecasting

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.model import ChatTime

dataset = "PTF"
hist_len = 120
pred_len = 24
model_path = "ChengsenWang/ChatTime-1-7B-Chat"

df = pd.read_csv(f"./dataset/{dataset}.csv")
hist_data = np.array(df["Hist"].apply(eval).values.tolist())[:, -hist_len:][0]
pred_data = np.array(df["Pred"].apply(eval).values.tolist())[:, :pred_len][0]
context = df["Text"].values[0]

model = ChatTime(hist_len=hist_len, pred_len=pred_len, model_path=model_path)

out_text = model.predict(hist_data, context)
out = model.predict(hist_data)

hist_x = np.linspace(0, hist_len-1, hist_len)
pred_x = np.linspace(hist_len, hist_len+pred_len-1, pred_len)

plt.figure(figsize=(8, 2), dpi=500)
plt.plot(hist_x, hist_data, color='#000000')
plt.plot(pred_x, pred_data, color='#000000', label='true')
plt.plot(pred_x, out_text, color='#FF7F0E', label='pred_text')
plt.plot(pred_x, out, color='#1F77B4', label='pred')
plt.axvline(hist_len, color='red')
plt.legend(loc="upper left")
plt.show()

```

### Time Series Question Answering

```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.model import ChatTime

dataset = "TSQA"
model_path = "ChengsenWang/ChatTime-1-7B-Chat"

df = pd.read_csv(f"./dataset/{dataset}.csv")
series = np.array(df["Series"].apply(eval).values.tolist())[0]
question = df["Question"].values[0]
answer = df["Answer"].values[0]

model = ChatTime(model_path=model_path)

out = model.analyze(question, series)

plt.figure(figsize=(8, 2), dpi=500)
plt.plot(series, color='#000000')
plt.show()

print(question)
print(f"\n{out} / {answer}\n")

```

### Continuous Pre-Training and Instruction Fine-Tuning

The code for Continuous Pre-Training and Instruction Fine-Tuning for ChatTime is located within the [training](./training/) folder, while the model weights are available at [ChengsenWang/ChatTime-1-7B-Base](https://huggingface.co/ChengsenWang/ChatTime-1-7B-Base) and [ChengsenWang/ChatTime-1-7B-Chat](https://huggingface.co/ChengsenWang/ChatTime-1-7B-Chat), respectively.

## :floppy_disk: Datasets

Refer to following repositories for instructions on downloading and utilizing the datasets.

- The datasets used in the ChatTime paper for Continuous Pre-Training and Instruction Fine-Tuning are available in the HuggingFace repositories: [ChengsenWang/ChatTime-1-Pretrain-1M](https://huggingface.co/datasets/ChengsenWang/ChatTime-1-Pretrain-1M) and [ChengsenWang/ChatTime-1-Finetune-100K](https://huggingface.co/datasets/ChengsenWang/ChatTime-1-Finetune-100K).
- Unimodal zero-shot forecasting datasets can be accessed via the previous [Google Drive](https://drive.google.com/drive/folders/1S7u4exc5NkKRWfdgqYBZ-VqSz9XfrEDV?usp=sharing), while multimodal datasets for context-guided forecasting and time series question-answering tasks are available on HuggingFace at [ChengsenWang/CGTSF](https://huggingface.co/datasets/ChengsenWang/CGTSF) and [ChengsenWang/TSQA](https://huggingface.co/datasets/ChengsenWang/TSQA).

## 📝 Citation

If you find this repo or our work useful for your research, please consider citing the [paper](https://arxiv.org/abs/2412.11376):

```tex
@inproceedings{
  author    = {Chengsen Wang and Qi Qi and Jingyu Wang and Haifeng Sun and Zirui Zhuang and Jinming Wu and Lei Zhang and Jianxin Liao},
  title     = {ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2025},
}
```

## 📪 Contact

If you have any question, please contact [cswang@bupt.edu.cn]().
