# CogVLM

## Introduction
CogVLM 是一个强大的开源视觉语言模型，利用视觉专家模块深度整合语言编码和视觉编码，在 14 项权威跨模态基准上取得了 SOTA 性能。目前仅支持英文，后续会提供中英双语版本支持，欢迎持续关注！

📖 [Paper（论文）](./assets/cogvlm-paper.pdf)

🌐 [web demo（测试网址）](http://36.103.203.44:7861/)

- CogVLM, a powerful open-source visual language foundation model. Different from the popular shallow-align method which maps image features into the input space of language model, **CogVLM bridges the gap between the frozen pretrained language model and image encoder by a trainable visual expert module in the attention and FFN layers**. CogVLM enables deep fusion of visual language features without sacrificing any performance on NLP tasks. 

- CogVLM-17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, and rank the 2nd on VQAv2, OKVQA, TextVQA, COCO captioning, etc., **surpassing or matching PaLI-X 55B**.

- We anticipate that the open-sourcing of CogVLM will greatly help the research and industrial application of visual understanding.

<div align="center">
    <img src=assets/metrics.png width=80% />
</div>

## Examples

CogVLM is powerful for answering various types of visual questions, including **Detailed Description & Visual Question Answering**,  **Complex Counting**, **Visual Math Problem Solving**, **OCR-Free Reasoning**, **OCR-Free Visual Question Answering**, **World Knowledge**, **Referring Expression Comprehension**, **Programming with Visual Input**, **Grounding with Caption**, **Grounding Visual Question Answering**, etc.

<div align="center">
    <img src=assets/compare.png width=80% />
</div>

<!-- ![compare](assets/compare.png) -->

<details>
<summary>Click to expand/collapse more examples</summary>

![Chat Examples](assets/chat.png)

</details>

## Method
CogVLM model comprises four fundamental components: a vision transformer (ViT) encoder, an MLP adapter, a pretrained large language model (GPT), and a visual expert module. See [Paper](./assets/cogvlm-paper.pdf) for more details.

<div align="center">
    <img src=assets/method.png width=70% />
</div>

## Usage

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Online Web Demo
We provide a [web demo](http://36.103.203.44:7861/) based on [Gradio](https://gradio.app).
<div align="center">
    <img src=assets/web_demo.png width=70% />
</div>

### Local Web Demo
We also offer a local web demo based on Gradio. First, install Gradio by running: `pip install gradio`. Then download and enter this repository and run `web_demo.py`. See the next section for detailed usage:

```bash
python web_demo.py --from_pretrained cogvlm-chat --version chat --english --bf16
python web_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16

```

### Terminal Demo
```bash
python cli_demo.py --from_pretrained cogvlm-base-224 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-base-490 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-chat --version chat --english --bf16
python cli_demo.py --from_pretrained cogvlm-grounding-base --version base --english --bf16
python cli_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
# We also support model parallel inference, which splits model to multiple (2/4/8) GPUs.
torchrun --standalone --nnodes=1 --nproc-per-node=2 cli_demo.py --from_pretrained cogvlm-chat --version chat --english --bf16
```

The program will automatically download the sat model and interact in the command line. You can generate replies by entering instructions and pressing enter. Enter 'clear' to clear the conversation history and 'stop' to stop the program.

Note:

* If you have trouble in accessing huggingface.co, you can add `--local_tokenizer /path/to/vicuna-7b-v1.5` to load the tokenizer.
* If you have trouble in automatically downloading model with 🔨[SAT](https://github.com/THUDM/SwissArmyTransformer), try downloading from 🤖[modelscope](https://www.modelscope.cn/models/ZhipuAI/CogVLM/summary) or 🤗[huggingface](https://huggingface.co/THUDM/CogVLM) manually.
* Download model using 🔨[SAT](https://github.com/THUDM/SwissArmyTransformer), the model will be saved to the default location `~/.sat_models`. Change the default location by setting the environment variable `SAT_HOME`. For example, if you want to save the model to `/path/to/my/models`, you can run `export SAT_HOME=/path/to/my/models` before running the python command.

The program provides the following hyperparameters to control the generation process:
```
usage: cli_demo.py [-h] [--max_length MAX_LENGTH] [--top_p TOP_P] [--top_k TOP_K] [--temperature TEMPERATURE] [--english]

optional arguments:
  -h, --help            show this help message and exit
  --max_length MAX_LENGTH
                        max length of the total sequence
  --top_p TOP_P         top p for nucleus sampling
  --top_k TOP_K         top k for top k sampling
  --temperature TEMPERATURE
                        temperature for sampling
  --english             only output English
```

### Fine-tuning

Start by downloading the [Captcha Images dataset](https://www.kaggle.com/datasets/aadhavvignesh/captcha-images). Once downloaded, extract the contents of the ZIP file.

To create a train/validation/test split in the ratio of 80/5/15, execute the following:

```bash
python scripts/split_dataset.py
```

Kickstart the fine-tuning process with this command:

```bash
bash scripts/finetune_(224/490)_lora.sh
```

Then, merge the model to model_parallel_size=1: (replace 4 with your training MP_SIZE)

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 merge_model.py --version base --bf16 --from_pretrained ./checkpoints/merged_lora_(224/490)
```

To evaluate the performance of your model, use:

```bash
bash scripts/evaluate_(224/490).sh
```

It is recommended to use 490 version. However, if you have limited GPU resources (such as only one node with eight 24GB 3090 cards), you can try 224 version with model parallel. The anticipated result is around 95% accuracy on test set. It is worth noting that the fine-tuning examples only tune limited parameters. If you want to improve performance, you can change trainable parameters in `finetune_demo.py` as needed.

## Model Quantization

Model quantization is not possible right now, but we are working on it. We will release the quantized model as soon as possible.

## License

The code in this repository is open source under the Apache-2.0 license, while the use of the CogVLM model weights must comply with the Model License.

## Citation & Acknowledgements

If you find our work helpful, please consider citing the following papers
```

```
In the instruction fine-tuning phase of the CogVLM, there are some English image-text data from the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLAVA](https://github.com/haotian-liu/LLaVA), [LRV-Instruction](https://github.com/FuxiaoLiu/LRV-Instruction), [LLaVAR](https://github.com/SALT-NLP/LLaVAR) and [Shikra](https://github.com/shikras/shikra) projects, as well as many classic cross-modal work datasets. We sincerely thank them for their contributions.
