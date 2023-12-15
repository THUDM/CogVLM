from dataclasses import dataclass
from enum import auto, Enum
from PIL.Image import Image
import requests
import re

import streamlit as st
from PIL import ImageDraw
from streamlit.delta_generator import DeltaGenerator


class Role(Enum):
    """
    CogVLM | CogAgent Only have 2 roles: USER, ASSISTANT
    """

    USER = auto()
    ASSISTANT = auto()

    def get_message(self):

        match self.value:
            case Role.USER.value:
                return st.chat_message(name="user", avatar="user")
            case Role.ASSISTANT.value:
                return st.chat_message(name="assistant", avatar="assistant")
            case _:
                st.error(f'Unexpected role: {self}')


@dataclass
class Conversation:
    role: Role = Role.USER
    content: str = ""
    image: Image | None = None
    content_show: str | None = None  # English translation of content

    def __str__(self) -> str:
        print(self.role, self.content)
        match self.role:
            case Role.USER | Role.ASSISTANT:
                return f'{self.role}\n{self.content}'

    def show(self, placeholder: DeltaGenerator | None = None) -> str:
        """
        show in markdown formate
        """
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()

        if self.role == Role.USER:
            if contains_chinese(self.content_show):
                self.content = translate_query(
                    "问：Translate the following texts into English: \n\n{}\n答：\n".format(self.content_show))
            else:
                self.content = self.content_show

            # self.content = translate_baidu(self.content_show, source_lan="zh", target_lan="en")
        if self.role == Role.ASSISTANT:
            self.content_show = translate_baidu(self.content, source_lan="en", target_lan="zh")
        message.markdown(self.content_show)
        if self.image:
            message.image(self.image)


def preprocess_text(history: list[Conversation], ) -> str:
    prompt = ""
    for conversation in history:
        prompt += f'{conversation}'
    prompt += f'{Role.ASSISTANT}\n'
    return prompt


def postprocess_text(template: str, text: str) -> str:
    """
    Replace <TASK> in the template with the given text.
    """
    quoted_text = f'"{text.strip()}"'
    return template.replace("<TASK>", quoted_text).strip()


def postprocess_image(text: str, img: Image) -> (str, Image):
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    pattern = r"\[\[(\d+),(\d+),(\d+),(\d+)\]\]"
    matches = re.findall(pattern, text)
    unique_matches = []
    draw = ImageDraw.Draw(img)
    if matches == []:
        return text, None

    for i, coords in enumerate(matches):
        if coords not in unique_matches:
            unique_matches.append(coords)
            scaled_coords = (
                int(float(coords[0]) * 0.001 * img.width),
                int(float(coords[1]) * 0.001 * img.height),
                int(float(coords[2]) * 0.001 * img.width),
                int(float(coords[3]) * 0.001 * img.height)
            )
            draw.rectangle(scaled_coords, outline=colors[i % len(colors)], width=3)
            color_text = f"(in {colors[i % len(colors)]} box)"
            text = text.replace(f"[[{','.join(coords)}]]", f"[[{','.join(coords)}]]{color_text}", 1)
    return text, img

# def postprocess_image(text: str, img: Image) -> (str, Image):
#     colors = ["red", "green", "blue", "yellow", "purple", "orange"]
#
#     pattern = r"\[\[(.*?)\]\]"
#     matches = re.findall(pattern, text)
#     if not matches:
#         return text, None
#     processed = set()
#     draw = ImageDraw.Draw(img)
#     for match in matches:
#         positions = match.group(1).split(';')
#         boxes = [tuple(map(int, pos.split(','))) for pos in positions if pos.replace(',', '').isdigit()]
#
#         for i, box in enumerate(boxes):
#             if box not in processed:  # 检查是否已经处理过这个坐标
#                 processed.add(box)  # 添加到已处理集合中
#
#                 # 将百分比坐标转换为实际像素坐标
#                 scaled_box = (
#                     int(box[0] * 0.001 * img.width),
#                     int(box[1] * 0.001 * img.height),
#                     int(box[2] * 0.001 * img.width),
#                     int(box[3] * 0.001 * img.height)
#                 )
#                 draw.rectangle(scaled_box, outline=colors[i % len(colors)], width=3)
#                 color_text = f"(in {colors[i % len(colors)]} box)"
#                 text = text.replace(f"[[{','.join(map(str, box))}]]", f"[[{','.join(map(str, box))}]]{color_text}", 1)
#
#     return text, img


def contains_chinese(text):
    for character in text:
        if '\u4e00' <= character <= '\u9fff':
            return True
    return False

def translate_query(query: str) -> str:
    return query
def translate_baidu(translate_text, source_lan, target_lan):
    return translate_text
