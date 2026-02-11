import random

import nodes
import server
from aiohttp import web
import os
import re
import json
import shutil
import yaml

from PIL import Image
from ..tools import (
    read_file_data,
    read_json_name,
    append_translate,
    return_file_value,
    baidutranslationapi,
    txt2img_system_content,
    call_openai,
    string_to_list,
    baidutranslation,
    clean_text,)

apikey_file = "set_apikey.json"   
prompt_builder_preset = {}

resource_path = os.path.join(os.path.dirname(__file__), "..", "json")
resource_path = os.path.abspath(resource_path)

try:
    pb_yaml_path = os.path.join(resource_path, 'prompt-builder.yaml')
    pb_yaml_path_example = os.path.join(resource_path, 'prompt-builder.yaml.example')

    if not os.path.exists(pb_yaml_path):
        shutil.copy(pb_yaml_path_example, pb_yaml_path)

    with open(pb_yaml_path, 'r', encoding="utf-8") as f:
        prompt_builder_preset = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print(f"[Inspire Pack] Failed to load 'prompt-builder.yaml'\nNOTE: Only files with UTF-8 encoding are supported.")


class PromptBuilderss:
    @classmethod
    def INPUT_TYPES(s):
        global prompt_builder_preset        
        apikey_name=read_json_name(read_file_data(apikey_file))
        presets = ["#PRESET"]
        return {"required": {                    
                        "model_name": (apikey_name,), 
                        "temperature_value": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}),
                        "category": (list(prompt_builder_preset.keys()) + ["#PLACEHOLDER"], ),
                        "preset": (presets, ),
                        "switch": ("BOOLEAN", {"default": True,"label_on": "中文", "label_off": "英文"},),
                        "text": ("STRING", {"multiline": True}),
                     },
                }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("CN","EN",)
    FUNCTION = "doit"

    CATEGORY = "Achen节点"

    def doit(self, **kwargs):
        text= kwargs['text']
        model_name= kwargs['model_name']
        temperature_value = kwargs['temperature_value']
        # 判断是否调用OpenAI函数扩写句子
        if model_name == "None":
            prompt_output_ch=text
            prompt_output_en=append_translate(text)
        else:
            system = txt2img_system_content
            url,key = return_file_value(model_name,apikey_file,value_a="url",value_b="key") 
            response_date=call_openai(key,url,text,model_name,system,temperature_value) # 调用openai

            # 调用函数将字符串转换为列表，并处理空行和小于64个字符的行
            filtered_list=string_to_list(response_date)
            prompt_output_ch = filtered_list[0]
            prompt_output_en = filtered_list[1] if len(filtered_list) > 1 else baidutranslationapi(prompt_output_ch,"en") 
        return (prompt_output_ch,prompt_output_en,)


@server.PromptServer.instance.routes.get("/node/prompt_builder")

def prompt_builder(request):
    result = {"presets": []}

    if "category" in request.rel_url.query:
        category = request.rel_url.query["category"]
        if category in prompt_builder_preset:
            result['presets'] = prompt_builder_preset[category]

    return web.json_response(result)


class PromptInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "text": ("STRING", {"multiline": True}),
                     },
                }

    RETURN_TYPES = ("STRING", )
    FUNCTION = "Promptin"

    CATEGORY = "Achen节点"

    def Promptin(self, **kwargs):
        text= kwargs['text']
        text_translate = append_translate(text)
        return (text_translate,)

NODE_CLASS_MAPPINGS = {
    "PromptBuilderss": PromptBuilderss,
    "PromptInput": PromptInput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptBuilderss": "构建提示词",
    "PromptInput": "提示词输入（中文）"
}

