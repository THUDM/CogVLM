# CogVLM

📖 [Paper（论文）](https://arxiv.org/abs/2311.03079)

🌐 [web demo（测试网址）](http://36.103.203.44:7861/)

🔥 **News**: ```2023/11/20``` We have updated the checkpoint, unified the versions of chat and VQA, and refreshed the SOTA on various datasets.

🔥 **News**: ```2023/11/20``` We release **[cogvlm-chat](https://huggingface.co/THUDM/cogvlm-chat-hf)**, **[cogvlm-grounding-generalist](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)/[base](https://huggingface.co/THUDM/cogvlm-grounding-base-hf)**, **[cogvlm-base-490](https://huggingface.co/THUDM/cogvlm-base-490-hf)/[224](https://huggingface.co/THUDM/cogvlm-base-224-hf)** on 🤗Huggingface. you can infer with transformers in [a few lines of code](#-transformers) now!

🔥 **News**: ```2023/10/27``` CogVLM bilingual version is available [online](https://chatglm.cn/)! Welcome to try it out!

[中文版README](./README_zh.md)

## Introduction
- CogVLM is a powerful **open-source visual language model** (**VLM**). CogVLM-17B has 10 billion vision parameters and 7 billion language parameters.

- CogVLM-17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, and rank the 2nd on VQAv2, OKVQA, TextVQA, COCO captioning, etc., **surpassing or matching PaLI-X 55B**. CogVLM can also [chat with you](http://36.103.203.44:7861) about images.

<div align="center">
    <img src=assets/metrics-min.png width=80% />
</div>

| Method           | LLM           | MM-VET | POPE(adversarial) | TouchStone |
| ---------------- | ------------- |--------| --------- |------------|
| BLIP-2           | Vicuna-13B    | 22.4   | -         | -          |
| Otter            | MPT-7B        | 24.7   | -         | -          |
| MiniGPT4         | Vicuna-13B    | 24.4   | 70.4      | 531.7      |
| InstructBLIP     | Vicuna-13B    | 25.6   | 77.3      | 552.4      |
| LLaMA-Adapter v2 | LLaMA-7B      | 31.4   | -         | 590.1      |
| LLaVA            | LLaMA2-7B     | 28.1   | 66.3      | 602.7      |
| mPLUG-Owl        | LLaMA-7B      | -      | 66.8      | 605.4      |
| LLaVA-1.5        | Vicuna-13B    | 36.3   | 84.5      | -          |
| Emu              | LLaMA-13B     | 36.3   | -         | -          |
| Qwen-VL-Chat     | -             | -      | -         | 645.2      |
| DreamLLM         | Vicuna-7B     | 35.9   | 76.5      | -          |
| CogVLM           | Vicuna-7B     | **52.8**   | **87.6**      | **742.0**      |

## Examples

<!-- CogVLM is powerful for answering various types of visual questions, including **Detailed Description & Visual Question Answering**,  **Complex Counting**, **Visual Math Problem Solving**, **OCR-Free Reasonging**, **OCR-Free Visual Question Answering**, **World Knowledge**, **Referring Expression Comprehension**, **Programming with Visual Input**, **Grounding with Caption**, **Grounding Visual Question Answering**, etc. -->
* CogVLM can accurately describe images in details with **very few hallucinations**.
    <details>
    <summary>Click for comparison with LLAVA-1.5 and MiniGPT-4.</summary>

    ![LLAVA Comparision](assets/llava-comparison-min.png)
    </details>
    <br>

* CogVLM can understand and answer various types of questions, and has a **visual grounding** version.
<div align="center">
    <img src=assets/pear_grounding.png width=90% />
</div>

<br>

* CogVLM sometimes captures more detailed content than GPT-4V(ision).
<div align="center">
    <img src=assets/compare-min.png width=90% />
</div>

<!-- ![compare](assets/compare.png) -->
<br> 

<details>
<summary>Click to expand more examples.</summary>

![Chat Examples](assets/chat.png)

</details>

## Method
CogVLM model comprises four fundamental components: a vision transformer (ViT) encoder, an MLP adapter, a pretrained large language model (GPT), and a **visual expert module**. See [Paper](./assets/cogvlm-paper.pdf) for more details.

<div align="center">
    <img src=assets/method-min.png width=70% />
</div>

## Get Started
We support two GUIs for model inference, **web demo** and **CLI**. If you want to use it in your python code, it is easy to modify the CLI scripts for your case. 

First, we need to install the dependencies.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

#### Hardware requirement
* Model Inference: 1 * A100(80G) or 2 * RTX 3090(24G).
* Finetuning: 4 * A100(80G) *[Recommend]* or 8* RTX 3090(24G).

<!-- ### Online Web Demo
We provide a [web demo](http://36.103.203.44:7861/) based on [Gradio](https://gradio.app). -->

### Web Demo
We also offer a local web demo based on Gradio. First, install Gradio by running: `pip install gradio`. Then download and enter this repository and run `web_demo.py`. See the next section for detailed usage:

```bash
python web_demo.py --from_pretrained cogvlm-chat-v1.1 --version chat --english --bf16
python web_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
```
The GUI of the web demo looks like:
<div align="center">
    <img src=assets/web_demo-min.png width=70% />
</div>

### CLI
We open-source different checkpoints for different downstreaming tasks:

* `cogvlm-chat-v1.1` The model supports multiple rounds of chat and vqa simultaneously, with different prompts.
* `cogvlm-base-224` The original checkpoint after text-image pretraining.
* `cogvlm-base-490` Amplify the resolution to 490 through position encoding interpolation from `cogvlm-base-224`.
* `cogvlm-grounding-generalist`. This checkpoint supports different visual grounding tasks, e.g. REC, Grounding Captioning, etc. 

Run CLI demo via:
```bash
# Chat version will provide detailed answers, while vqa version usually only has one word in answer.
python cli_demo.py --from_pretrained cogvlm-base-224 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-base-490 --version base --english --bf16 --no_prompt
python cli_demo.py --from_pretrained cogvlm-chat-v1.1 --version chat --english --bf16
python cli_demo.py --from_pretrained cogvlm-chat-v1.1 --version vqa --english --bf16
python cli_demo.py --from_pretrained cogvlm-grounding-generalist --version base --english --bf16
```
The program will automatically download the sat model and interact in the command line. You can generate replies by entering instructions and pressing enter.
Enter `clear` to clear the conversation history and `stop` to stop the program.

#### Multi-GPU inference
We also support model parallel inference, which splits model to multiple (2/4/8) GPUs. `--nproc-per-node=[n]` in the following command controls the number of used GPUs.
```
torchrun --standalone --nnodes=1 --nproc-per-node=2 cli_demo.py --from_pretrained cogvlm-chat-v1.1 --version chat --english --bf16
```

**Note**:

* If you have trouble in accessing huggingface.co, you can add `--local_tokenizer /path/to/vicuna-7b-v1.5` to load the tokenizer.
* If you have trouble in automatically downloading model with 🔨[SAT](https://github.com/THUDM/SwissArmyTransformer), try downloading from 🤖[modelscope](https://www.modelscope.cn/models/ZhipuAI/CogVLM/summary) or 🤗[huggingface](https://huggingface.co/THUDM/CogVLM) or 💡[wisemodel](https://www.wisemodel.cn/models/ZhipuAI/CogVLM) manually.
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

### Finetuning
You may want to use CogVLM in your own task, which needs a **different output style or domain knowledge**. We here provide a finetuning example for **Captcha Recognition**.

1. Start by downloading the [Captcha Images dataset](https://www.kaggle.com/datasets/aadhavvignesh/captcha-images). Once downloaded, extract the contents of the ZIP file.

2. To create a train/validation/test split in the ratio of 80/5/15, execute the following:
    ```bash
    python scripts/split_dataset.py
    ```

3. Start the fine-tuning process with this command:

    ```bash
    bash scripts/finetune_(224/490)_lora.sh
    ```

4. Merge the model to `model_parallel_size=1`: (replace the 4 below with your training `MP_SIZE`)

    ```bash
    torchrun --standalone --nnodes=1 --nproc-per-node=4 merge_model.py --version base --bf16 --from_pretrained ./checkpoints/merged_lora_(224/490)
    ```

5. Evaluate the performance of your model.
    ```bash
    bash scripts/evaluate_(224/490).sh
    ```

It is recommended to use the `490px` version. However, if you have limited GPU resources (such as only one node with 8* RTX 3090), you can try `224px` version with model parallel. 

The anticipated result of this script is around `95%` accuracy on test set.

It is worth noting that the fine-tuning examples only tune limited parameters. (Expert only) If you want to get `>98%` accuracy, you need to increase the trainable parameters in `finetune_demo.py`.


### 🤗 Transformers

To use CogVLM for the inference with transformers, use the following code:

```python
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()


# chat example

query = 'Describe this image'
image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true', stream=True).raw).convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))

# This image captures a moment from a basketball game. Two players are prominently featured: one wearing a yellow jersey with the number
# 24 and the word 'Lakers' written on it, and the other wearing a navy blue jersey with the word 'Washington' and the number 34. The player
# in yellow is holding a basketball and appears to be dribbling it, while the player in navy blue is reaching out with his arm, possibly
# trying to block or defend. The background shows a filled stadium with spectators, indicating that this is a professional game.</s>



# vqa example

query = 'How many houses are there in this cartoon?'
image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/3.jpg?raw=true', stream=True).raw).convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='vqa')   # vqa mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))

# 4</s>
```

### OpenAI Vision formate

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

## License

The code in this repository is open source under the [Apache-2.0 license](./LICENSE), while the use of the CogVLM model weights must comply with the [Model License](./MODEL_LICENSE).

## Citation & Acknowledgements

If you find our work helpful, please consider citing the following papers
```
@article{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
In the instruction fine-tuning phase of the CogVLM, there are some English image-text data from the [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLAVA](https://github.com/haotian-liu/LLaVA), [LRV-Instruction](https://github.com/FuxiaoLiu/LRV-Instruction), [LLaVAR](https://github.com/SALT-NLP/LLaVAR) and [Shikra](https://github.com/shikras/shikra) projects, as well as many classic cross-modal work datasets. We sincerely thank them for their contributions.
