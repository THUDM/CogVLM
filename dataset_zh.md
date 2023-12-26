# CogVLM-SFT-311K：CogVLM SFT 中的双语视觉指令数据集

CogVLM-SFT-311K 是我们在训练 **CogVLM v1.0** 最初版本时使用的主要对齐语料库。此数据集的构建过程如下：
1. 从开源的 [MiniGPT-4](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) 中选取了大约3500个高质量数据样本，称为 minigpt4-3500。
2. 将 minigpt4-3500 与 [Llava-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) 整合，并通过语言模型翻译获得中文部分。
3. 我们发现在 minigpt4-3500 和 Llava-instruct 的详细描述部分存在许多噪声。因此，我们纠正了这两部分的中文语料，并将纠正后的语料重新翻译成英语。

## 许可证
+ 由于非商业协议限制，我们没有在 CogVLM的双语版本 和其他任何 涉及商业化的模型 中使用这些数据。 
+ 数据集许可证遵守：<br> Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use
这将不允许你使用这些数据进行任何 **商业化行为**。

## 数据集地址

+ [CogVLM-SFT-311K](https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K)

## 数据集信息
数据集共有三个文件夹，分别对应混合 minigpt4-3500 与llava混合的一部分数据集，llava 单论对话和多轮对话数据集。其布局如下：
```
.CogVLM-SFT-311K
├── llava_details-minigpt4_3500_formate
├── llava_instruction_multi_conversations_formate
└── llava_instruction_single_conversation_formate
```
在我们开源的数据中，数据集按照以下格式分布
```
.llava_details-minigpt4_3500_formate
├── images
│   └── 00000001.jpg
└── labels
    └──  00000001.json
```
其中，images存储图像数据，而labels存储这张图像对应的描述或对话。

## 数据集数量
+ llava_details-minigpt4_3500_formate 22464 张图片和描述
+ llava_instruction_muti_conversations_formate 56673 张图片和多轮连续对话
+ llava_instruction_single_conversation_formate 76634 张图片和单轮对话
## 数据集格式

图像描述 Caption 格式
```
{
  "captions": [
    {
      "role": "caption",
      "content": "这张照片展示了一男一女，他们都穿着红嘿色衣服，坐在桌子旁，在参加活动时面带微笑。这对情侣正在一起摆姿势，背景中还有其他人。在整个场景中可以看到另外几个人，一些人坐在附近的桌子旁，另一些人坐得离这对夫妇更近。各种餐桌设置，如酒杯和杯子，被放置在桌子上。排在最前面的男子打着领带。"
    }
  ]
}
```
图像对话 Json 格式
``` 
{
    "conversations": [
      {
        "role": "user",
        "content": "在尝试保持这个浴室有序时可能会遇到哪些挑战？",
      },
      {
        "role": "assistant",
        "content": "在尝试保持这空间和各种物品的存在，例如杯子和手机占用了可用的空间。在图像中，浴室柜台上只有一个水槽可用，这导致多个用户可能必须共享空间放置自己的物品。这可能导致杂乱和混乱外的储物解决方案，如架子、橱柜或墙壁挂架，以保持浴室有序并最小化柜台上的杂乱。"
      },
    ]
}
```

## References
This project utilizes data and concepts based on the following research papers:
- Zhu, D., Chen, J., Shen, X., Li, X., & Elhoseiny, M. (2023). MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. arXiv preprint arXiv:2304.10592.
- Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. arXiv:2304.08485.