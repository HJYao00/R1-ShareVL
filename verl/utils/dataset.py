# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
import random


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def add_gaussian_noise(self, image, mean=0, std=10):
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(mean, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def horizontal_flip(self, pil_img):
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT)

    def vertical_flip(self, pil_img):
        return pil_img.transpose(Image.FLIP_TOP_BOTTOM)

    def horizontal_vertical_flip(self, pil_img):
        return pil_img.transpose(Image.ROTATE_180)

    def process_image(self, image: Union[Dict[str, Any], ImageObject],transform=None) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        if transform == 'rotate_90cw':
            image = image.rotate(-90, expand=True) 
        elif transform == 'rotate_90ccw':
            image = image.rotate(90, expand=True)
        elif transform == 'rotate_180cw':
            image = image.rotate(180, expand=True)  
        elif transform == 'rotate_180ccw':
            image = image.rotate(-180, expand=True) 
        elif transform == 'gaussian_noise_mean0_std10':
            image = self.add_gaussian_noise(image, mean=0, std=10)  
        elif transform == 'gaussian_noise_mean0_std25':
            image = self.add_gaussian_noise(image, mean=0, std=25) 
        elif transform == 'gaussian_noise_mean0_std50':
            image = self.add_gaussian_noise(image, mean=0, std=50) 
        elif transform == 'vertically_flipped':
            image = self.vertical_flip(image)
        elif transform =='horizontally_flipped':
            image = self.horizontal_flip(image)
        elif transform =='horizontally_vertically_flipped':
            image = self.horizontal_vertical_flip(image)
        
        return image

class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        question_num: int = None
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.question_num = question_num

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:  # remote dataset
            self.dataset = load_dataset(data_path, split=data_split)
        
        self.transformations = {
            'rotate_90cw': "The image is rotated 90 degrees counterclockwise. Please rotate it back and help me solve the question based on the corrected image.\n",  
            'rotate_90ccw': "The image is rotated 90 degrees clockwise. Please rotate it back and help me solve the question based on the corrected image.",  
            'rotate_180cw': "The image is rotated 180 degrees counterclockwise. Please rotate it back and help me solve the question based on the corrected image.\n",  
            # 'rotate_180ccw': "Please rotate the image 180 degrees clockwise.", 
            'gaussian_noise_mean0_std10': "This image has been corrupted with noise. Please solve the question based on this image.\n",  
            'gaussian_noise_mean0_std25': "This image has been corrupted with noise. Please solve the question based on this image.\n", 
            'gaussian_noise_mean0_std50': "This image has been corrupted with noise. Please solve the question based on this image.\n", 
            'vertically_flipped': 'The image is vertically flipped. Please flip it back to its original orientation and help me solve the question based on the corrected image.\n',
            'horizontally_flipped': 'The image is horizontally flipped. Please flip it back to its original orientation and help me solve the question based on the corrected image.\n',
            'horizontally_vertically_flipped': 'The image is flipped both horizontally and vertically, which is equivalent to a 180-degree rotation. Please restore it to its original orientation and help me solve the question based on the corrected image.\n'
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row_dict: dict = self.dataset[index]
        prompt_str: str = row_dict[self.prompt_key]
        question_list = []
        if "new_questions" in row_dict and isinstance(row_dict["new_questions"], str):
            row_dict["new_questions"] = eval(row_dict["new_questions"])
        if "new_questions" in row_dict:
            row_dict["new_questions"].append(prompt_str)
            selected_questions = random.sample(row_dict["new_questions"], self.question_num)
            question_list.extend(selected_questions)

        else:
            question_list.append(prompt_str)
        if self.format_prompt:
            question_list_format = [q + " " + self.format_prompt.strip() for q in question_list]

        input_ids_list = [] 
        attention_mask_list = []
        position_ids_list = []
        prompt_list = []
        if self.image_key in row_dict:
            row_dict["multi_modal_inputs"] = []
            row_dict["multi_modal_data"] = []

        for question in question_list_format:
            if random.random() < 0.3:
                transform_image_method = random.choice(list(self.transformations.keys()))
                transform_prompt = self.transformations[transform_image_method]
            else:
                transform_image_method = None
                transform_prompt = ""
            if self.image_key in row_dict:
                # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
                content_list = []
                for i, content in enumerate(question.split("<image>")):
                    if i != 0:
                        content_list.append({"type": "image"})
                    if content:
                        content_list.append({"type": "text", "text": transform_prompt + content})
                messages = [{"role": "user", "content": content_list}]
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt_list.append(prompt)
                images = [self.process_image(image,transform_image_method) for image in row_dict[self.image_key]]
                model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

                row_dict["multi_modal_data"].append({"image": images})
                row_dict["multi_modal_inputs"].append(dict(model_inputs))
            else:
                messages = [{"role": "user", "content": question}]
                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                prompt_list.append(prompt)
                model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

            if self.processor is not None and (self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor" or self.processor.image_processor.__class__.__name__ == 'Qwen2VLImageProcessorFast'):
                position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    attention_mask=attention_mask,
                ) 
            else:
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None) 

            input_ids, attention_mask, position_ids = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
        if self.image_key in row_dict:
            _ = row_dict.pop(self.image_key)
        row_dict["input_ids"] = torch.stack(input_ids_list, dim=0)
        row_dict["attention_mask"] = torch.stack(attention_mask_list,dim=0)
        row_dict["position_ids"] = torch.stack(position_ids_list,dim=0)
        row_dict["raw_prompt_ids"] = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompt_list]
        row_dict["ground_truth"] = row_dict.pop(self.answer_key)
        return row_dict
