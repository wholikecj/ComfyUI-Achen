from openai import OpenAI
import json
import os
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import hashlib
import inspect
from server import PromptServer
import folder_paths
from aiohttp import web
import re
from ..tools import (
    read_file_data,
    read_json_name,
    return_file_value,
    call_openai,
    baidutranslationapi,
    append_translate,
    txt2img_system_content,
    clean_text,)

case_file = "case_text.json"
apikey_file = "set_apikey.json"
magic_options = "magic_options.json"

   
# 读取一个 JSON 文件的内容并返回。 确保内容与预期格式匹配。
def read_json_file(file_path):
    
    if not os.access(file_path, os.R_OK):
        print(f"Warning: No read permissions for file {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
            # Check if the content matches the expected format.
            if not all(['name' in item and 'prompt' in item and 'negative_prompt' in item for item in content]):
                print(f"Warning: Invalid content in file {file_path}")
                return None
            return content
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {str(e)}")
        return None

# 从提供的JSON数据中返回样式名称。
def read_sdxl_styles(json_data):
   
    if not isinstance(json_data, list):
        print("Error: input data must be a list")
        return []

    return [item['name'] for item in json_data if isinstance(item, dict) and 'name' in item]

# 返回目录中styles子目录的所有JSON文件。
def get_all_json_files(directory):
       subdirectory = os.path.join(directory, 'styles')
       return [os.path.join(subdirectory, file) for file in os.listdir(subdirectory) if file.endswith('.json') and os.path.isfile(os.path.join(subdirectory, file))]

# 从目录中的所有 JSON 文件加载样式。通过添加后缀重命名重复的样式名称。
def load_styles_from_directory(subdirectory):
    
    json_files = get_all_json_files(subdirectory)
    combined_data = []
    seen = set()

    for json_file in json_files:
        json_data = read_json_file(json_file)
        if json_data:
            for item in json_data:
                original_style = item['name']
                style = original_style
                suffix = 1
                while style in seen:
                    style = f"{original_style}_{suffix}"
                    suffix += 1
                item['name'] = style
                seen.add(style)
                combined_data.append(item)

    unique_style_names = [item['name'] for item in combined_data if isinstance(item, dict) and 'name' in item]

    return combined_data, unique_style_names


def validate_json_data(json_data):
    """
    Validates the structure of the JSON data.
    """
    if not isinstance(json_data, list):
        return False
    for template in json_data:
        if 'name' not in template or 'prompt' not in template:
            return False
    return True

def find_template_by_name(json_data, template_name):
    """
    Returns a template from the JSON data by name or None if not found.
    """
    for template in json_data:
        if template['name'] == template_name:
            return template
    return None

def replace_prompts_in_template(template, positive_prompt, negative_prompt):
    """
   使用提供的提示替换给定模板中的占位符。

    参数：
    - template (dict)：包含提示占位符的模板。
    - positive_prompt (str)：用于替换模板中“{prompt}”的正向提示。
    - negative_prompt (str)：要与模板中任何现有负向提示合并的负向提示。

    返回：
    - tuple：包含替换后的正向和负向提示的元组。
    """
    positive_result = template['prompt'].replace('{prompt}', positive_prompt)

    json_negative_prompt = template.get('negative_prompt', "")
    negative_result = f"{json_negative_prompt}, {negative_prompt}" if json_negative_prompt and negative_prompt else json_negative_prompt or negative_prompt

    return positive_result, negative_result

def deduplicate_merge(prompt_1, prompt_2):
    """
    合并两个提示，对标记进行去重。

    参数：
    - prompt1 (str)：第一个提示。
    - prompt2 (str)：第二个提示。

    返回：
    - str：合并并去重后的提示。
    """
    if not prompt_2:
        return prompt_1
    elif not prompt_1:
        return prompt_2

    token_prompt_1 = list(map(lambda x: x.strip(), prompt_1.split(",")))
    token_prompt_2 = list(map(lambda x: x.strip(), prompt_2.split(",")))

    # deduplicate common prompt parts
    for token in token_prompt_1:
        if token in token_prompt_2:
            token_prompt_2.remove(token)

    token_prompt_1.extend(token_prompt_2)

    prompt_out = ", ".join(token_prompt_1)

    return prompt_out

def read_sdxl_templates_replace_and_combine(json_data, template_name, positive_prompt, negative_prompt):
    """
    通过名称查找特定模板，然后用提供的提示替换并组合其占位符。

    参数：
    - json_data (list)：模板列表。
    - template_name (str)：所需模板的名称。
    - positive_prompt (str)：用于替换占位符的正向提示。
    - negative_prompt (str)：要组合的负向提示。

    返回：
    - tuple：包含替换并组合后的正向和负向提示的元组。
    """
    if not validate_json_data(json_data):
        return positive_prompt, negative_prompt

    template = find_template_by_name(json_data, template_name)

    if template:
        return replace_prompts_in_template(template, positive_prompt, negative_prompt)
    else:
        return positive_prompt, negative_prompt



@PromptServer.instance.routes.get("/preview/{name}")
async def view(request):
    name = request.match_info["name"]

    image_path = name
    filename = os.path.basename(image_path)
    return web.FileResponse(image_path, headers={"Content-Disposition": f"filename=\"{filename}\""})


def populate_items(styles, item_type):
    for idx, item_name in enumerate(styles):
        current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        preview_path = os.path.join(current_directory, item_type, item_name + ".png")

        if len(item_name.split('-')) > 1:
            content = f"{item_name.split('-')[0]} /{item_name}"
        else:
            content = item_name

        if os.path.exists(preview_path):
            styles[idx] = {
                "content": content,
                "preview": preview_path
            }
        else:
            # print(f"Warning: Preview image '{item_name}.png' not found for item '{item_name}'")
            styles[idx] = {
                "content": content,
                "preview": None
            }
def text_positive_llam(key,url,model_name,text_positive,max_tokens,system_content):
    client = OpenAI(api_key=key, base_url=url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": text_positive},
        ],
        max_tokens=max_tokens,
        stream=False
    )
    original_response = response.choices[0].message.content
    processed_response = re.sub(r'"', '', original_response)
    return processed_response




class PromptStyler:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        current_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.json_data, styles = load_styles_from_directory(current_directory)
        apikey_name=read_json_name(read_file_data(apikey_file))
        types = {
            "required": {
                "style": ((styles), ), 
                "model_name": (apikey_name,),                 
                "max_temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}), 
                "text_positive": ("STRING", {"multiline": True,"default": "小猫", }),                
                "text_negative": ("STRING", {"multiline": True,"default": "", }), 
                #"switch": ("BOOLEAN", {"default": False},)
            },
        }
        style = types["required"]["style"][0]
        populate_items(style, "images")
        return types

    RETURN_TYPES = ('STRING','STRING',)
    RETURN_NAMES = ('正面','负面',)
    FUNCTION = 'prompt_styler'
    CATEGORY = 'Achen节点'

    def prompt_styler(self, text_positive, text_negative, style,max_temperature, model_name ): 

        style = style["content"].split("/")[-1]
        text_positive_styled, text_negative_styled = read_sdxl_templates_replace_and_combine(self.json_data, style, text_positive, text_negative) 

        # 判断是否调用OpenAI函数扩写句子
        if model_name == "None":
            text_positive_styled=append_translate(text_positive_styled)
        else:
            system = txt2img_system_content
            url,key = return_file_value(model_name,apikey_file,value_a="url",value_b="key")            
            text_positive_styled=call_openai(key,url,text_positive_styled,model_name,system,max_temperature)        
        
        return clean_text(text_positive_styled), clean_text(text_negative_styled)

import json
import os

class PhotoPrompt:
    @classmethod
    def INPUT_TYPES(s):
        # 从JSON文件加载选项       
        options = s.load_options()  
        return {
            "optional": {
		        "Subject": ("STRING", {"multiline": True,"default": "主体,细节,场景",}),
                "film": (list(options["胶片"].keys()),),
                "Photographer": (list(options["摄影师"].keys()),),
                "composition_shot": (list(options["构图镜头"].keys()),),
                "camera": (list(options["相机"].keys()),),
                "color_grading": (list(options["色调"].keys()),),
                "time_of_day": (list(options["拍摄时间"].keys()),),
                "lighting": (list(options["灯光"].keys()),)
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('prompt',)
    FUNCTION = "generate_prompt"
    CATEGORY = "Achen节点"

    @staticmethod
    def load_options():
        # 获取当前脚本所在的目录的父目录
        current_dir = os.path.dirname(os.path.dirname(__file__))
        json_path = os.path.join(current_dir,"json" ,"magic_options.json")
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def generate_prompt(self, Subject,Photographer, camera, composition_shot, time_of_day,film, color_grading, lighting):
        options = self.load_options()        
        
        # 连接每个选定选项的提示
        prompt_parts = [
        part for part in [
            Subject,
            options["胶片"].get(film, ""),
            options["摄影师"].get(Photographer, ""),
            options["构图镜头"].get(composition_shot, ""),
            options["相机"].get(camera, ""),
            options["色调"].get(color_grading, ""),
            options["拍摄时间"].get(time_of_day, ""),
            options["灯光"].get(lighting, ""),
        ] if part.strip() != ""
        ]
        # 用逗号连接提示的各个部分
        final_prompt = ", ".join(prompt_parts)        
        return (final_prompt,)

class PaintingPrompt:
    @classmethod
    def INPUT_TYPES(s):
        apikey_name=read_json_name(read_file_data(apikey_file))
        # Load options from JSON file        
        options = s.load_options()  
        return {
            "optional": {
		        "Subject": ("STRING", {"multiline": True,"default": "主体,细节,场景",}),
                "Artist": (list(options["艺术家"].keys()),),
                "Style": (list(options["风格"].keys()),),
                "Composition": (list(options["构图"].keys()),),
                "Emotion": (list(options["情感"].keys()),),
                "Color": (list(options["颜色"].keys()),)
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('prompt',)    
    FUNCTION = "generate_prompt"
    CATEGORY = "Achen节点"

    @staticmethod
    def load_options():
        # 获取当前脚本所在的目录的父目录
        current_dir = os.path.dirname(os.path.dirname(__file__))
        json_path = os.path.join(current_dir,"json" ,"magic_options.json")
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def generate_prompt(self, Subject, Style, Composition, Emotion, Color, Artist):
        options = self.load_options()
        
        # Concatenate the prompts for each selected option
        prompt_parts = [
        part for part in [            
            options["艺术家"].get(Artist, ""),
            options["风格"].get(Style, ""),
            Subject,
            options["构图"].get(Composition, ""),
            options["情感"].get(Emotion, ""),
            options["颜色"].get(Color, ""),
        ] if part.strip() != ""
    ]
        # Join the prompt parts with commas
        final_prompt = ", ".join(prompt_parts)
        
        return (final_prompt,)


NODE_CLASS_MAPPINGS = {
    #"PromptStyler": PromptStyler,
    "PhotoPrompt": PhotoPrompt,
    "PaintingPrompt": PaintingPrompt
}


# 一个包含节点友好/可读的标题的字典
NODE_DISPLAY_NAME_MAPPINGS = {
   # "PromptStyler": "风格提示词",
    "PhotoPrompt": "照片提示词",
    "PaintingPrompt": "绘画提示词"
}

