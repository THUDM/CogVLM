# CogVLM-SFT-311K: Bilingual Visual Instruction Data in CogVLM SFT

CogVLM-SFT-311K is the primary aligned corpus used in the initial training of CogVLM v1.0. The process of constructing this dataset is as follows:
1. Approximately 3500 high-quality data samples were selected from the open source [MiniGPT-4](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align), known as minigpt4-3500.
2. Minigpt4-3500 was integrated with [Llava-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and translated into Chinese through a language model.
3. We discovered significant noise in the detailed description part of minigpt4-3500 and Llava-instruct. Thus, we corrected these Chinese corpora and retranslated them into English.

## License

+ Due to non-commercial agreements, we did not use these data in the bilingual version of CogVLM or any other models involving commercialization.
+ The dataset license adheres to: <br> Attribution-NonCommercial 4.0 International. It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use
This will not allow you to use these data for any **commercial activitiesI**.

## Dataset Address

+ [CogVLM-SFT-311K](https://huggingface.co/datasets/THUDM/CogVLM-SFT-311K)

## Dataset Information

The dataset contains three folders corresponding to the mixed part of minigpt4-3500 and llava, the llava solo conversation, and the multi-turn conversation datasets. Their layout is as follows:
```
.CogVLM-SFT-311K
├── llava_details-minigpt4_3500_formate
├── llava_instruction_multi_conversations_formate
└── llava_instruction_single_conversation_formate
```
In our open-source data, the datasets are distributed as follows:
```
.llava_details-minigpt4_3500_formate
├── images
│   └── 00000001.jpg
└── labels
    └── 00000001.json
```
Images are stored in the images folder, while labels containing the corresponding image description or dialogue are stored in the labels folder.

## Dataset Quantity

+ llava_details-minigpt4_3500_formate: 22,464 images and descriptions
+ llava_instruction_multi_conversations_formate: 56,673 images and multi-turn conversations
+ llava_instruction_single_conversation_formate: 76,634 images and single-turn conversations

Dataset Format

Caption format for image description
```
{
  "captions": [
    {
      "role": "caption",
      "content": "The photograph features a beach scene with a group of people surfing in the ocean. There are ten individuals spaced out in the water, sitting or lying on various sizes of body boards. The surfers line up in a row, close to the sandy shoreline. Some of the body boards are visible floating on the surface of the water, with the surfers riding on them or nearby."
    }
  ]
}
```
Conversation format for image dialogue
```
{
  "conversations": [
    {
      "role": "user",
      "content": "What can be inferred about the zebras' behavior and surroundings?"
    },
    {
      "role": "assistant",
      "content": "Based on the image, we can infer that the two zebras are likely seeking relief from the sun's heat, as they are standing side by side under the branches of a thorny tree. This shade-providing tree offers some respite from the sun, possibly during the hottest part of the day. The zebras are in a green field with grass, providing them with an ideal environment to graze and eat while staying near their source of shelter. This shows that the zebras' behavior is influenced by the conditions and available resources in their surroundings. It also highlights that these animals adopt strategies to adapt to the fluctuating conditions of their environment, such as cooperation and seeking shelter, to survive and thrive in their natural habitat."
    }
  ]
}
```

## References
This project utilizes data and concepts based on the following research papers:
- Zhu, D., Chen, J., Shen, X., Li, X., & Elhoseiny, M. (2023). MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. arXiv preprint arXiv:2304.10592.
- Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. arXiv:2304.08485. 