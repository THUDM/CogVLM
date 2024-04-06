"""
This script is designed to mimic the OpenAI API interface with CogVLM & CogAgent Chat
It demonstrates how to integrate image and text-based input to generate a response.
Currently, the model can only handle a single image.
Therefore, do not use this script to process multiple images in one conversation. (includes images from history)
And it only works on the chat model, not the base model.
"""
import requests
import json
import base64

base_url = "http://127.0.0.1:8000"


def create_chat_completion(model, messages, temperature=0.8, max_tokens=2048, top_p=0.8, use_stream=False):
    """
    This function sends a request to the chat API to generate a response based on the given messages.

    Args:
        model (str): The name of the model to use for generating the response.
        messages (list): A list of message dictionaries representing the conversation history.
        temperature (float): Controls randomness in response generation. Higher values lead to more random responses.
        max_tokens (int): The maximum length of the generated response.
        top_p (float): Controls diversity of response by filtering less likely options.
        use_stream (bool): Determines whether to use a streaming response or a single response.

    The function constructs a JSON payload with the specified parameters and sends a POST request to the API.
    It then handles the response, either as a stream (for ongoing responses) or a single message.
    """

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
    """
    Encodes an image file into a base64 string.
    Args:
        image_path (str): The path to the image file.

    This function opens the specified image file, reads its content, and encodes it into a base64 string.
    The base64 encoding is used to send images over HTTP as text.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def simple_image_chat(use_stream=True, img_path=None):
    """
    Facilitates a simple chat interaction involving an image.

    Args:
        use_stream (bool): Specifies whether to use streaming for chat responses.
        img_path (str): Path to the image file to be included in the chat.

    This function encodes the specified image and constructs a predefined conversation involving the image.
    It then calls `create_chat_completion` to generate a response from the model.
    The conversation includes asking about the content of the image and a follow-up question.
    """

    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_url
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
    ]
    create_chat_completion("cogvlm-chat-17b", messages=messages, use_stream=use_stream)


if __name__ == "__main__":
    simple_image_chat(use_stream=False, img_path="demo.jpg")
