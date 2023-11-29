"""
This script is designed to mimic the OpenAI API interface to interact with the cogvlm-17B
It demonstrates the integration of image and text-based inputs for generating responses.
Currently, this model can only process a single image and does not have the ability to process image history.
And it is only for chat model, not base model.
"""
import requests
import json
import base64

base_url = "http://127.0.0.1:8000"


def create_chat_completion(model, messages, temperature=0.8, max_tokens=2048, top_p=0.8, use_stream=False):
    data = {
        "model": model,
        "messages": messages,
        "stream": use_stream,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=use_stream)
    if response.status_code == 200:
        if use_stream:
            # 处理流式响应
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')[6:]
                    try:
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content)
                    except:
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应
            decoded_line = response.json()
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            print(content)
    else:
        print("Error:", response.status_code)
        return None


def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def simple_image_chat(use_stream=True, img_path=None):
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"

    # img_url_before if for this demo only.
    img_url_before = f"data:image/jpeg;base64,{encode_image('demo_summer.jpg')}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": "What are they  in these image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url_before
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": "The image displays a wooden boardwalk extending through a vibrant green grassy wetland. The sky is partly cloudy with soft, wispy clouds, indicating nice weather. Vegetation is seen on either side of the boardwalk, and trees are present in the background, suggesting that this area might be a natural reserve or park designed for ecological preservation and outdoor recreation. The boardwalk allows visitors to explore the area without disturbing the natural habitat.",
        },
        {
            "role": "user",
            "content": "Do you think this is a spring or winter photo?"
        },
        {
            "role": "assistant",
            "content": "Based on the lush greenery, the absence of snow, and the overall vibrancy of the scene, it appears to be a spring or summer photo.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What about this? this photo is about winter,right?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
                    },
                },
            ],
        },
    ]
    create_chat_completion("cogvlm-chat-17b", messages=messages, use_stream=use_stream)


if __name__ == "__main__":
    simple_image_chat(use_stream=False, img_path="demo_winter.jpg")
