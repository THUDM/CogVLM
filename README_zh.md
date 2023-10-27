# CogVLM

📖 [Paper（论文）](./assets/cogvlm-paper.pdf)

🌐 [web demo（测试网址）](http://36.103.203.44:7861/)

🔥 **News**: CogVLM bilingual version is available [online](https://chatglm.cn/)! Welcome to try it out!

🔥 **News**: CogVLM中英双语版正式[上线](https://chatglm.cn/)了！欢迎体验！

[README in English](./README.md)

## 简介
- CogVLM 是一个强大的开源视觉语言模型（VLM）。CogVLM-17B 拥有 100 亿视觉参数和 70 亿语言参数。

- CogVLM-17B 在 10 个经典跨模态基准测试上取得了 SOTA 性能，包括 NoCaps、Flicker30k captioning、RefCOCO、RefCOCO+、RefCOCOg、Visual7W、GQA、ScienceQA、VizWiz VQA 和 TDIUC，而在 VQAv2、OKVQA、TextVQA、COCO captioning 等方面则排名第二，超越或与 PaLI-X 55B 持平。您可以通过线上 [demo](http://36.103.203.44:7861) 体验 CogVLM 多模态对话。

<div align="center">
    <img src=assets/metrics-min.png width=80% />
</div>

## 示例

<!-- CogVLM is powerful for answering various types of visual questions, including **Detailed Description & Visual Question Answering**,  **Complex Counting**, **Visual Math Problem Solving**, **OCR-Free Reasonging**, **OCR-Free Visual Question Answering**, **World Knowledge**, **Referring Expression Comprehension**, **Programming with Visual Input**, **Grounding with Caption**, **Grounding Visual Question Answering**, etc. -->
* CogVLM 能够准确地描述图像，**几乎不会出现幻觉**。
    <details>
    <summary>点击查看与 LLAVA-1.5 和 MiniGPT-4 的比较。</summary>

    ![LLAVA Comparision](assets/llava-comparison-min.png)
    </details>
<br>

* CogVLM 能理解和回答各种类型的问题，并有一个**视觉定位**版本。
<div align="center">
    <img src=assets/pear_grounding.png width=90% />
</div>

<br>

* CogVLM 有时比 GPT-4V(ision) 提取到更多的细节信息。
<div align="center">
    <img src=assets/compare-min.png width=90% />
</div>

<!-- ![compare](assets/compare.png) -->
<br> 

<details>
<summary>点击展开更多示例。</summary>

![Chat Examples](assets/chat.png)

</details>

## 方法
CogVLM 模型包括四个基本组件：视觉变换器（ViT）编码器、MLP适配器、预训练的大型语言模型（GPT）和一个**视觉专家模块**。更多细节请参见[论文](./assets/cogvlm-paper.pdf)。

<div align="center">
    <img src=assets/method-min.png width=70% />
</div>

## 入门指南
我们提供两种图形用户界面（GUI）进行模型推断，分别是**网页演示**和**命令行界面（CLI）**。如果您想在Python代码中使用它，很容易修改CLI脚本以适应您的情况。

首先，需要安装依赖项。

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### 硬件要求
* 模型推断：1 * A100(80G) 或 2 * RTX 3090(24G)。
* 微调：4 * A100(80G) [推荐] 或 8 * RTX 3090(24G)。

<!-- ### Online Web Demo
We provide a [web demo](http://36.103.203.44:7861/) based on [Gradio](https://gradio.app). -->

### 网页演示
我们还提供基于Gradio的本地网页演示。首先，通过运行 pip install gradio 安装Gradio。然后下载并进入此仓库，运行 web_demo.py。具体使用方式如下：

```bash
python web_demo.py --from_pretrained cogvlm-chat --version chat --english --bf16
python web_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
```
网页演示的 GUI 界面如下：
<div align="center">
    <img src=assets/web_demo-min.png width=70% />
</div>

### CLI
我们开源了不同下游任务的模型权重：

* cogvlm-chat 用于对齐的模型，在此之后支持像 GPT-4V 一样的聊天。
* cogvlm-base-224 文本-图像预训练后的原始权重。
* cogvlm-base-490 从 cogvlm-base-224 微调得到的 490px 分辨率版本。
* cogvlm-grounding-generalist 这个权重支持不同的视觉定位任务，例如 REC、Grounding Captioning 等。

通过CLI演示，执行以下命令：
```bash
python cli_demo.py --from_pretrained cogvlm-base-224 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-base-490 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-chat --version chat --english --bf16
python cli_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
```
该程序会自动下载 sat 模型并在命令行中进行交互。您可以通过输入指令并按 Enter 生成回复。
输入 clear 可清除对话历史，输入 stop 可停止程序。


## 许可

此存储库中的代码是根据 [Apache-2.0 许可](./LICENSE) 开放源码，而使用 CogVLM 模型权重必须遵循 [模型许可](./MODEL_LICENSE)。

## 引用 & 鸣谢

如果您觉得我们的工作有帮助，请考虑引用以下论文：
```

```
在 CogVLM 的指令微调阶段，我们使用了来自 [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) 、 [LLAVA](https://github.com/haotian-liu/LLaVA) 、 [LRV-Instruction](https://github.com/FuxiaoLiu/LRV-Instruction)、 [LLaVAR](https://github.com/SALT-NLP/LLaVAR) 和 [Shikra](https://github.com/shikras/shikra) 项目的一些英文图像-文本数据，以及许多经典的跨模态工作数据集。我们衷心感谢他们的贡献。
