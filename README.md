# CogVLM & CogAgent

📗 [中文版README](./README_zh.md)

🌟 **Jump to detailed introduction: [Introduction to CogVLM](#introduction-to-cogvlm)，
🆕 [Introduction to CogAgent](#introduction-to-cogagent)**

📔 For more detailed usage information, please refer to: [CogVLM & CogAgent's technical documentation (in Chinese)](https://zhipu-ai.feishu.cn/wiki/LXQIwqo1OiIVTykMh9Lc3w1Fn7g) 

<table>
  <tr>
    <td>
      <h2> CogVLM </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2311.03079">CogVLM: Visual Expert for Pretrained Language Models</a></p>
      <p><b>CogVLM</b> is a powerful open-source visual language model (VLM). CogVLM-17B has 10 billion visual parameters and 7 billion language parameters, <b>supporting image understanding and multi-turn dialogue with a resolution of 490*490</b>.</p>
      <p><b>CogVLM-17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks</b>, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC.</p>
    </td>
    <td>
      <h2> CogAgent </h2>
      <p> 📖  Paper: <a href="https://arxiv.org/abs/2312.08914">CogAgent: A Visual Language Model for GUI Agents </a></p>
      <p><b>CogAgent</b> is an open-source visual language model improved based on CogVLM. CogAgent-18B has 11 billion visual parameters and 7 billion language parameters, <b>supporting image understanding at a resolution of 1120*1120</b>. <b>On top of the capabilities of CogVLM, it further possesses GUI image Agent capabilities</b>.</p>
      <p> <b>CogAgent-18B achieves state-of-the-art generalist performance on 9 classic cross-modal benchmarks</b>, including VQAv2, OK-VQ, TextVQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE. <b>It significantly surpasses existing models on GUI operation datasets</b> including AITW and Mind2Web.</p>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <p>🌐 Web Demo for both CogVLM and CogAgent: <a href="http://36.103.203.44:7861">this link</a></p>
    </td>
  </tr>
</table>


**Table of Contents**

- [CogVLM \& CogAgent](#cogvlm--cogagent)
    - [Release](#release)
    - [Get Started](#get-started)
        - [Option 1: Inference Using Web Demo.](#option-1-inference-using-web-demo)
        - [Option 2：Deploy CogVLM / CogAgent by yourself](#option-2deploy-cogvlm--cogagent-by-yourself)
            - [Situation 2.1 CLI (SAT version)](#situation-21-cli-sat-version)
            - [Situation 2.2 CLI (Huggingface version)](#situation-22-cli-huggingface-version)
            - [Situation 2.3 Web Demo](#situation-23-web-demo)
        - [Option 3：Finetuning CogAgent / CogVLM](#option-3finetuning-cogagent--cogvlm)
        - [Option 4: OpenAI Vision format](#option-4-openai-vision-format)
        - [Hardware requirement](#hardware-requirement)
        - [Model checkpoints](#model-checkpoints)
    - [Introduction to CogVLM](#introduction-to-cogvlm)
        - [Examples](#examples)
    - [Introduction to CogAgent](#introduction-to-cogagent)
        - [GUI Agent Examples](#gui-agent-examples)
    - [Cookbook](#cookbook)
        - [Task Prompts](#task-prompts)
        - [Which --version to use](#which---version-to-use)
        - [FAQ](#faq)
    - [License](#license)
    - [Citation \& Acknowledgements](#citation--acknowledgements)

## Release
- 🔥🔥🔥  **News**: ```2024/4/5```: [CogAgent](https://arxiv.org/abs/2312.08914) was selected as a CVPR 2024 Highlights!
- 🔥🔥  **News**: ```2023/12/26```: We have released the [CogVLM-SFT-311K](dataset.md) dataset, 
  which contains over 150,000 pieces of data that we used for **CogVLM v1.0 only** training. Welcome to follow and use.
- 🔥 **News**: ```2023/12/18```: **New Web UI Launched!** We have launched a new web UI based on Streamlit,
  users can painlessly talk to CogVLM, CogAgent in our UI. Have a better user experience.
- **News**: ```2023/12/15```: **CogAgent Officially Launched!** CogAgent is an image understanding model developed
  based on CogVLM. It features **visual-based GUI Agent capabilities** and has further enhancements in image
  understanding. It supports image input with a resolution of 1120*1120, and possesses multiple abilities including
  multi-turn dialogue with images, GUI Agent, Grounding, and more.

- **News**: ```2023/12/8``` We have updated the checkpoint of cogvlm-grounding-generalist to
  cogvlm-grounding-generalist-v1.1, with image augmentation during training, therefore more robust.
  See [details](#introduction-to-cogvlm).

- **News**: ```2023/12/7``` CogVLM supports **4-bit quantization** now! You can inference with just **11GB** GPU memory!

- **News**: ```2023/11/20``` We have updated the checkpoint of cogvlm-chat to cogvlm-chat-v1.1, unified the versions of
  chat and VQA, and refreshed the SOTA on various datasets. See [details](#introduction-to-cogvlm)

- **News**: ```2023/11/20``` We release **[cogvlm-chat](https://huggingface.co/THUDM/cogvlm-chat-hf)**, **[cogvlm-grounding-generalist](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)/[base](https://huggingface.co/THUDM/cogvlm-grounding-base-hf)**, **[cogvlm-base-490](https://huggingface.co/THUDM/cogvlm-base-490-hf)/[224](https://huggingface.co/THUDM/cogvlm-base-224-hf)** on 🤗Huggingface. you can infer with transformers in [a few lines of code](#situation-22-cli-huggingface-version)now!

- ```2023/10/27``` CogVLM bilingual version is available [online](https://chatglm.cn/)! Welcome to try it out!

- ```2023/10/5``` CogVLM-17B released。

## Get Started

### Option 1: Inference Using Web Demo.

* Click here to enter [CogVLM & CogAgent Web Demo](http://36.103.203.44:7861/)。

If you need to use Agent and Grounding functions, please refer to [Cookbook - Task Prompts](#task-prompts)

### Option 2：Deploy CogVLM / CogAgent by yourself

We support two GUIs for model inference, **CLI** and **web demo** . If you want to use it in your python code, it is
easy to modify the CLI scripts for your case.


<!-- ### Online Web Demo
We provide a [web demo](http://36.103.203.44:7861/) based on [Gradio](https://gradio.app). -->


First, we need to install the dependencies.

```bash
# CUDA >= 11.8
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**All code for inference is located under the ``basic_demo/`` directory. Please switch to this directory first before
proceeding with further operations.**

#### Situation 2.1 CLI (SAT version)

Run CLI demo via:

```bash
# CogAgent
python cli_demo_sat.py --from_pretrained cogagent-chat --version chat --bf16  --stream_chat
python cli_demo_sat.py --from_pretrained cogagent-vqa --version chat_old --bf16  --stream_chat

# CogVLM
python cli_demo_sat.py --from_pretrained cogvlm-chat --version chat_old --bf16  --stream_chat
python cli_demo_sat.py --from_pretrained cogvlm-grounding-generalist --version base --bf16  --stream_chat
```

The program will automatically download the sat model and interact in the command line. You can generate replies by
entering instructions and pressing enter.
Enter `clear` to clear the conversation history and `stop` to stop the program.

We also support model parallel inference, which splits model to multiple (2/4/8) GPUs. `--nproc-per-node=[n]` in the
following command controls the number of used GPUs.

```
torchrun --standalone --nnodes=1 --nproc-per-node=2 cli_demo_sat.py --from_pretrained cogagent-chat --version chat --bf16
```

- If you want to manually download the weights, you can replace the path after ``--from_pretrained`` with the model
  path.

- Our model supports SAT's **4-bit quantization** and **8-bit quantization**.
  You can change ``--bf16`` to ``--fp16``, or ``--fp16 --quant 4``, or ``--fp16 --quant 8``.

  For example

    ```bash
    python cli_demo_sat.py --from_pretrained cogagent-chat --fp16 --quant 8 --stream_chat
    python cli_demo_sat.py --from_pretrained cogvlm-chat-v1.1 --fp16 --quant 4 --stream_chat
    # In SAT version，--quant should be used with --fp16
    ```

- The program provides the following hyperparameters to control the generation process:
    ```
    usage: cli_demo_sat.py [-h] [--max_length MAX_LENGTH] [--top_p TOP_P] [--top_k TOP_K] [--temperature TEMPERATURE]

    optional arguments:
    -h, --help            show this help message and exit
    --max_length MAX_LENGTH
                            max length of the total sequence
    --top_p TOP_P         top p for nucleus sampling
    --top_k TOP_K         top k for top k sampling
    --temperature TEMPERATURE
                            temperature for sampling
    ```

- Click [here](#which---version-to-use) to view the correspondence between different models and the ``--version``
  parameter.

#### Situation 2.2 CLI (Huggingface version)

Run CLI demo via:

```bash
# CogAgent
python cli_demo_hf.py --from_pretrained THUDM/cogagent-chat-hf --bf16
python cli_demo_hf.py --from_pretrained THUDM/cogagent-vqa-hf --bf16

# CogVLM
python cli_demo_hf.py --from_pretrained THUDM/cogvlm-chat-hf --bf16
python cli_demo_hf.py --from_pretrained THUDM/cogvlm-grounding-generalist-hf --bf16
```

- If you want to manually download the weights, you can replace the path after ``--from_pretrained`` with the model
  path.

- You can change ``--bf16`` to ``--fp16``, or ``--quant 4``. For example, our model supports Huggingface's **4-bit
  quantization**:

    ```bash
    python cli_demo_hf.py --from_pretrained THUDM/cogvlm-chat-hf --quant 4
    ```

#### Situation 2.3 Web Demo

We also offer a local web demo based on Gradio. First, install Gradio by running: `pip install gradio`. Then download
and enter this repository and run `web_demo.py`. See the next section for detailed usage:

```bash
python web_demo.py --from_pretrained cogagent-chat --version chat --bf16
python web_demo.py --from_pretrained cogagent-vqa --version chat_old --bf16
python web_demo.py --from_pretrained cogvlm-chat-v1.1 --version chat_old --bf16
python web_demo.py --from_pretrained cogvlm-grounding-generalist --version base --bf16
```

The GUI of the web demo looks like:

<div align="center">
    <img src=assets/web_demo-min.png width=70% />
</div>

### Option 3：Finetuning CogAgent / CogVLM

You may want to use CogVLM in your own task, which needs a **different output style or domain knowledge**. **All code
for finetuning is located under the ``finetune_demo/`` directory.**

We here provide a finetuning example for **Captcha Recognition** using lora.

1. Start by downloading the [Captcha Images dataset](https://www.kaggle.com/datasets/aadhavvignesh/captcha-images). Once
   downloaded, extract the contents of the ZIP file.

2. To create a train/validation/test split in the ratio of 80/5/15, execute the following:
    ```bash
    python utils/split_dataset.py
    ```

3. Start the fine-tuning process with this command:

    ```bash
    bash finetune_demo/finetune_(cogagent/cogvlm)_lora.sh
    ```

4. Merge the model to `model_parallel_size=1`: (replace the 4 below with your training `MP_SIZE`)

    ```bash
    torchrun --standalone --nnodes=1 --nproc-per-node=4 utils/merge_model.py --version base --bf16 --from_pretrained ./checkpoints/merged_lora_(cogagent/cogvlm490/cogvlm224)
    ```

5. Evaluate the performance of your model.
    ```bash
    bash finetune_demo/evaluate_(cogagent/cogvlm).sh
    ```

### Option 4: OpenAI Vision format

We provide the same API examples as `GPT-4V`, which you can view in `openai_demo`.

1. First, start the node

```
python openai_demo/openai_api.py
```

2. Next, run the request example node, which is an example of a continuous dialogue

```
python openai_demo/openai_api_request.py
```

3. You will get output similar to the following

```
This image showcases a tranquil natural scene with a wooden pathway leading through a field of lush green grass. In the distance, there are trees and some scattered structures, possibly houses or small buildings. The sky is clear with a few scattered clouds, suggesting a bright and sunny day.
```

### Hardware requirement

* Model Inference:

  For INT4 quantization: 1 * RTX 3090(24G)   (CogAgent takes ~ 12.6GB, CogVLM takes ~ 11GB)

  For FP16: 1 * A100(80G) or 2 * RTX 3090(24G)

* Finetuning:

  For FP16: 4 * A100(80G) *[Recommend]* or 8* RTX 3090(24G).

### Model checkpoints

If you run the `basic_demo/cli_demo*.py` from the code repository, it will automatically download SAT or Hugging Face
weights. Alternatively, you can choose to manually download the necessary weights.

- CogAgent

  |   Model name    | Input resolution |                             Introduction                             | Huggingface model | SAT model |
  | :-----------: | :----: | :----------------------------------------------------------: | :------: | :-------: |
  | cogagent-chat |  1120  | Chat version of CogAgent. Supports GUI Agent, multiple-round  chat and visual grounding. |  [HF link](https://huggingface.co/THUDM/cogagent-chat-hf) <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/cogagent-chat-hf)    |   [HF link](https://huggingface.co/THUDM/CogAgent/tree/main)<br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/CogAgent)           |
  | cogagent-vqa |  1120  | VQA version of CogAgent. Has stronger capabilities in single-turn visual dialogue. Recommended for VQA benchmarks. |  [HF link](https://huggingface.co/THUDM/cogagent-vqa-hf)<br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/cogagent-vqa-hf)        |    [HF link](https://huggingface.co/THUDM/CogAgent/tree/main) <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/CogAgent)      |
c
- CogVLM

  |          Model name           | Input resolution |                           Introduction                            | Huggingface model | SAT model |
  | :-------------------------: | :----: | :-------------------------------------------------------: | :------: | :-------: |
  |         cogvlm-chat-v1.1         |  490   |  Supports multiple rounds of chat and vqa simultaneously, with different prompts.   |  [HF link](https://huggingface.co/THUDM/cogvlm-chat-hf) <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/cogvlm-chat-hf)        |    [HF link](https://huggingface.co/THUDM/CogVLM/tree/main)  <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/CogVLM)       |
  |       cogvlm-base-224       |  224   |               The original checkpoint after text-image pretraining.               |   [HF link](https://huggingface.co/THUDM/cogvlm-base-224-hf) <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/cogvlm-base-224-hf)       |     [HF link](https://huggingface.co/THUDM/CogVLM/tree/main) <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/CogVLM)       |
  |       cogvlm-base-490       |  490   |      Amplify the resolution to 490 through position encoding interpolation from `cogvlm-base-224`.      |   [HF link](https://huggingface.co/THUDM/cogvlm-base-490-hf) <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/cogvlm-base-490-hf)       |     [HF link](https://huggingface.co/THUDM/CogVLM/tree/main) <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/CogVLM)       |
  | cogvlm-grounding-generalist |  490   | This checkpoint supports different visual grounding tasks, e.g. REC, Grounding Captioning, etc.  |    [HF link](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)  <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/cogvlm-grounding-generalist-hf)       |     [HF link](https://huggingface.co/THUDM/CogVLM/tree/main)   <br> [OpenXLab link](https://openxlab.org.cn/models/detail/THUDM/CogVLM)     |

## Introduction to CogVLM

- CogVLM is a powerful **open-source visual language model** (**VLM**). CogVLM-17B has 10 billion vision parameters and
  7 billion language parameters.

- CogVLM-17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks, including NoCaps, Flicker30k
  captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, and rank the 2nd on VQAv2,
  OKVQA, TextVQA, COCO captioning, etc., **surpassing or matching PaLI-X 55B**. CogVLM can
  also [chat with you](http://36.103.203.44:7861) about images.

<div align="center">
    <img src=assets/metrics-min.png width=50% />
</div>

<details>
<summary>Click to view results on MM-VET, POPE, TouchStone. </summary>

<table>
    <tr>
        <td>Method</td>
        <td>LLM</td>
        <td>MM-VET</td>
        <td>POPE(adversarial)</td>
        <td>TouchStone</td>
    </tr>
    <tr>
        <td>BLIP-2</td>
        <td>Vicuna-13B</td>
        <td>22.4</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Otter</td>
        <td>MPT-7B</td>
        <td>24.7</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>MiniGPT4</td>
        <td>Vicuna-13B</td>
        <td>24.4</td>
        <td>70.4</td>
        <td>531.7</td>
    </tr>
    <tr>
        <td>InstructBLIP</td>
        <td>Vicuna-13B</td>
        <td>25.6</td>
        <td>77.3</td>
        <td>552.4</td>
    </tr>
    <tr>
        <td>LLaMA-Adapter v2</td>
        <td>LLaMA-7B</td>
        <td>31.4</td>
        <td>-</td>
        <td>590.1</td>
    </tr>
    <tr>
        <td>LLaVA</td>
        <td>LLaMA2-7B</td>
        <td>28.1</td>
        <td>66.3</td>
        <td>602.7</td>
    </tr>
    <tr>
        <td>mPLUG-Owl</td>
        <td>LLaMA-7B</td>
        <td>-</td>
        <td>66.8</td>
        <td>605.4</td>
    </tr>
    <tr>
        <td>LLaVA-1.5</td>
        <td>Vicuna-13B</td>
        <td>36.3</td>
        <td>84.5</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Emu</td>
        <td>LLaMA-13B</td>
        <td>36.3</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>Qwen-VL-Chat</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>645.2</td>
    </tr>
    <tr>
        <td>DreamLLM</td>
        <td>Vicuna-7B</td>
        <td>35.9</td>
        <td>76.5</td>
        <td>-</td>
    </tr>
    <tr>
        <td>CogVLM</td>
        <td>Vicuna-7B</td>
        <td> <b>52.8</b> </td>
        <td><b>87.6</b></td>
        <td><b>742.0</b></td>
    </tr>
</table>

</details>

<details>
<summary>Click to view results of cogvlm-grounding-generalist-v1.1. </summary>

<table>
    <tr>
        <td></td>
        <td>RefCOCO</td>
        <td></td>
        <td></td>
        <td>RefCOCO+</td>
        <td></td>
        <td></td>
        <td>RefCOCOg</td>
        <td></td>
        <td>Visual7W</td>
    </tr>
    <tr>
        <td></td>
        <td>val</td>
        <td>testA</td>
        <td>testB</td>
        <td>val</td>
        <td>testA</td>
        <td>testB</td>
        <td>val</td>
        <td>test</td>
        <td>test</td>
    </tr>
    <tr>
        <td>cogvim-grounding-generalist</td>
        <td>92.51</td>
        <td>93.95</td>
        <td>88.73</td>
        <td>87.52</td>
        <td>91.81</td>
        <td>81.43</td>
        <td>89.46</td>
        <td>90.09</td>
        <td>90.96</td>
    </tr>
    <tr>
        <td>cogvim-grounding-generalist-v1.1</td>
        <td>**92.76**</td>
        <td>**94.75**</td>
        <td>**88.99**</td>
        <td>**88.68**</td>
        <td>**92.91**</td>
        <td>**83.39**</td>
        <td>**89.75**</td>
        <td>**90.79**</td>
        <td>**91.05**</td>
    </tr>
</table>
</details>

### Examples

<!-- CogVLM is powerful for answering various types of visual questions, including **Detailed Description & Visual Question Answering**,  **Complex Counting**, **Visual Math Problem Solving**, **OCR-Free Reasonging**, **OCR-Free Visual Question Answering**, **World Knowledge**, **Referring Expression Comprehension**, **Programming with Visual Input**, **Grounding with Caption**, **Grounding Visual Question Answering**, etc. -->

* CogVLM can accurately describe images in details with **very few hallucinations**.
    <details>
    <summary>Click for comparison with LLAVA-1.5 and MiniGPT-4.</summary>

    <img src=assets/llava-comparison-min.png width=50% />

    </details>
    <br>

* CogVLM can understand and answer various types of questions, and has a **visual grounding** version.

<div align="center">
    <img src=assets/pear_grounding.png width=50% />
</div>

<br>

* CogVLM sometimes captures more detailed content than GPT-4V(ision).

<div align="center">
    <img src=assets/compare-min.png width=50% />
</div>

<!-- ![compare](assets/compare.png) -->
<br> 

<details>
<summary>Click to expand more examples.</summary>

![Chat Examples](assets/chat.png)

</details>

## Introduction to CogAgent

CogAgent is an open-source visual language model improved based on CogVLM. CogAgent-18B has 11 billion visual parameters
and 7 billion language parameters

CogAgent-18B achieves state-of-the-art generalist performance on 9 classic cross-modal benchmarks, including VQAv2,
OK-VQ, TextVQA, ST-VQA, ChartQA, infoVQA, DocVQA, MM-Vet, and POPE. It significantly surpasses existing models on GUI
operation datasets such as AITW and Mind2Web.

In addition to all the features already present in CogVLM (visual multi-round dialogue, visual grounding), CogAgent:

1. Supports higher resolution visual input and dialogue question-answering. **It supports ultra-high-resolution image
   inputs of 1120x1120.**

2. **Possesses the capabilities of a visual Agent**, being able to return a plan, next action, and specific operations
   with coordinates for any given task on any GUI screenshot.

3. **Enhanced GUI-related question-answering capabilities**, allowing it to handle questions about any GUI screenshot,
   such as web pages, PC apps, mobile applications, etc.

4. Enhanced capabilities in OCR-related tasks through improved pre-training and fine-tuning.

<div align="center">
    <img src=assets/cogagent_function.jpg width=60% />
</div>

### GUI Agent Examples

<div align="center">
    <img src=assets/cogagent_main_demo.jpg width=90% />
</div>

## Cookbook

### Task Prompts

1. **General Multi-Round Dialogue**: Say whatever you want.

2. **GUI Agent Task**: Use the [Agent template](https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L761)
   and replace \<TASK\> with the task instruction enclosed in double quotes. This query can make CogAgent infer Plan and
   Next Action. If adding ``(with grounding)`` at the end of the query, the model will return a formalized action
   representation with coordinates.

For example, to ask the model how to complete the task "Search for CogVLM" on a current GUI screenshot, follow these
steps:

1. Randomly select a template from
   the [Agent template](https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L761). Here, we
   choose ``What steps do I need to take to <TASK>?``.

2. Replace <TASK> with the task instruction enclosed in double quotes, for
   example, ``What steps do I need to take to "Search for CogVLM"?`` . Inputting this to the model yields:

> Plan: 1. Type 'CogVLM' into the Google search bar. 2. Review the search results that appear. 3. Click on a relevant
> result to read more about CogVLM or access further resources.
>
> Next Action: Move the cursor to the Google search bar, and type 'CogVLM' into it.

3. If adding ``(with grounding)`` at the end, i.e. changing the input
   to ``What steps do I need to take to "Search for CogVLM"?(with grounding)``, the output of CogAgent would be:

> Plan: 1. Type 'CogVLM' into the Google search bar. 2. Review the search results that appear. 3. Click on a relevant
> result to read more about CogVLM or access further resources.
>
> Next Action: Move the cursor to the Google search bar, and type 'CogVLM' into it.
> Grounded Operation:[combobox] Search -> TYPE: CogVLM at the box [[212,498,787,564]]

Tip: For GUI Agent tasks, it is recommended to conduct only single-round dialogues for each image for better results.

3. **Visual Grounding**. Three modes of grounding are supported:

    - Image description with grounding coordinates (bounding box). Use any template
      from [caption_with_box template](https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L537) as model
      input. For example:

   > Can you provide a description of the image and include the coordinates [[x0,y0,x1,y1]] for each mentioned object?

    - Returning grounding coordinates (bounding box) based on the description of objects. Use any template
      from [caption2box template](https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L345),
      replacing ``<expr>`` with the object's description. For example:

   > Can you point out *children in blue T-shirts* in the image and provide the bounding boxes of their location?

    - Providing a description based on bounding box coordinates. Use a template
      from [box2caption template](https://github.com/THUDM/CogVLM/blob/main/utils/utils/template.py#L400),
      replacing ``<objs>`` with the position coordinates. For example:

   > Tell me what you see within the designated area *[[086,540,400,760]]* in the picture.

**Format of coordination:** The bounding box coordinates in the model's input and output use the
format ``[[x1, y1, x2, y2]]``, with the origin at the top left corner, the x-axis to the right, and the y-axis
downward. (x1, y1) and (x2, y2) are the top-left and bottom-right corners, respectively, with values as relative
coordinates multiplied by 1000 (prefixed with zeros to three digits).

### Which --version to use

Due to differences in model functionalities, different model versions may have distinct ``--version`` specifications for
the text processor, meaning the format of the prompts used varies.

|         model name          | --version |
|:---------------------------:|:---------:|
|        cogagent-chat        |   chat    |
|        cogagent-vqa         | chat_old  |
|         cogvlm-chat         | chat_old  |
|      cogvlm-chat-v1.1       | chat_old  |
| cogvlm-grounding-generalist |   base    |
|       cogvlm-base-224       |   base    |
|       cogvlm-base-490       |   base    |

### FAQ

* If you have trouble in accessing huggingface.co, you can add `--local_tokenizer /path/to/vicuna-7b-v1.5` to load the
  tokenizer.
* If you have trouble in automatically downloading model with 🔨[SAT](https://github.com/THUDM/SwissArmyTransformer), try
  downloading from 🤖[modelscope](https://www.modelscope.cn/models/ZhipuAI/CogVLM/summary) or
  🤗[huggingface](https://huggingface.co/THUDM/CogVLM) or 💡[wisemodel](https://www.wisemodel.cn/models/ZhipuAI/CogVLM)
  manually.
* Download model using 🔨[SAT](https://github.com/THUDM/SwissArmyTransformer), the model will be saved to the default
  location `~/.sat_models`. Change the default location by setting the environment variable `SAT_HOME`. For example, if
  you want to save the model to `/path/to/my/models`, you can run `export SAT_HOME=/path/to/my/models` before running
  the python command.

## License

The code in this repository is open source under the [Apache-2.0 license](./LICENSE), while the use of the CogVLM model
weights must comply with the [Model License](./MODEL_LICENSE).

## Citation & Acknowledgements

If you find our work helpful, please consider citing the following papers

```
@misc{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{hong2023cogagent,
      title={CogAgent: A Visual Language Model for GUI Agents}, 
      author={Wenyi Hong and Weihan Wang and Qingsong Lv and Jiazheng Xu and Wenmeng Yu and Junhui Ji and Yan Wang and Zihan Wang and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2312.08914},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```

In the instruction fine-tuning phase of the CogVLM, there are some English image-text data from
the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLAVA](https://github.com/haotian-liu/LLaVA), [LRV-Instruction](https://github.com/FuxiaoLiu/LRV-Instruction), [LLaVAR](https://github.com/SALT-NLP/LLaVAR)
and [Shikra](https://github.com/shikras/shikra) projects, as well as many classic cross-modal work datasets. We
sincerely thank them for their contributions.
