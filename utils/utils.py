import base64
import json
from typing import Dict, List
import logging
import os
from PIL import Image

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def encode_image(image_path):
    """Encode an image as base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error in file {file_path}: {e}")
    return data


def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            
# process current batch and generate a 2D image_list
# the outer list corresponds to the length of prompts, and the inner list contains all images corresponding to each Prompt
def process_batch(current_batch):
    """
    args current_batch:
        like [ [message_dict_1, message_dict_2, ...],  # the first Prompt
               [message_dict_1, message_dict_2, ...],  # the second Prompt
               ...
             ]
    return:
        like [ [Image_1, Image_2, ...],               # images corresponding to the first Prompt
               [Image_1, Image_2, ...],               # images corresponding to the second Prompt
               ...
             ]
    """
    batched_images = []

    for messages in current_batch:
        images_for_this_prompt = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for element in content:
                    if element.get("type") == "image":
                        image_path = element.get("image")
                        if image_path and os.path.exists(image_path):
                            image = Image.open(image_path)
                            images_for_this_prompt.append(image)
        batched_images.append(images_for_this_prompt)

    return batched_images


def remove_image_fields(messages):
    """
    for each message in the messages list:
        - if message['content'] is not a list, keep it as is.
        - if message['content'] is a list, iterate over its elements:
            * if element is a dict and element['type'] == 'image',
                remove the 'image' field from the dict, but keep the dict itself (with "type": "image").
            * otherwise, leave it as is.
    """
    updated_messages = []
    for message in messages:
        content = message.get("content", [])

        # if content is not a list, keep it as is
        if not isinstance(content, list):
            updated_messages.append(message)
            continue

        updated_content = []
        for element in content:
            # if element is a dict and element['type'] == 'image', remove the 'image' field from the dict
            if isinstance(element, dict) and element.get("type") == "image":
                new_element = dict(element)
                new_element.pop("image", None)
                updated_content.append(new_element)
            else:
                updated_content.append(element)

        # replace the content in message
        updated_message = {**message, "content": updated_content}
        updated_messages.append(updated_message)

    return updated_messages            
