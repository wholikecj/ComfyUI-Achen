import torch
import re
import base64
import json
import os
import pickle
import urllib
import urllib.request
import urllib.parse
import zlib
import folder_paths
import uuid
import numpy as np
from ..tools import (
    read_file_data,
    read_json_name,
    append_translate,
    return_file_value,
    baidutranslationapi, 
    baidutranslation,
    string_to_list,
    clean_text,)


start_file = "base_start.json"


BASE_RESOLUTIONS = [
    (512, 512),
    (512, 682),
    (512, 768),    
    (512, 910),
    (512, 1024),
    (768, 768),
    (768, 910),
    (768, 1024),
    (768, 1280),
    (768, 1344),
    (768, 1536),
    (816, 1920),
    (832, 1152),
    (896, 1088),
    (896, 1152),
    (1024, 1024),
    (1080, 1920),
    (1440, 2560),
    #("自定义", "自定义"),
]


class Latent_nodes:
    resolution_strings = [f"{width} x {height}" for width, height in BASE_RESOLUTIONS]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (cls.resolution_strings, {"default": "512 x 768"}),
                #"width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                #"height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "swap_dimensions": ("BOOLEAN", {"default": False},),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
            }
        }
    RETURN_TYPES = ("LATENT",  )
    RETURN_NAMES = ( "latent", )
    FUNCTION = "Latent_nodes"
    CATEGORY = "Achen节点"

    def Latent_nodes(self, resolution,swap_dimensions,batch_size):
        
        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width_in, height_in = map(int, resolution.split(' x '))
                width = width_in
                height = height_in
            except ValueError:
                raise ValueError("基础分辨率格式无效。")
        
        if swap_dimensions == True:
            width, height = height, width
        
        width = int(width)
        height = int(height)
        
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
           
        return( {"samples":latent},  )    



class ClipTextCNEncode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        types = {
            "required": {
                "CLIP": ("CLIP", {"tooltip": "用于对文本进行编码的CLIP模型。"}),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "要编码的文本。"}), 
            },
        }
        return types

    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'prompt_text'
    CATEGORY = 'Achen节点'

    def prompt_text(self,CLIP,text): 
        # 调用百度api翻译
        text=append_translate(text)
        # 清理符号
        text=clean_text(text)
         
        tokens = CLIP.tokenize(text)
        output = CLIP.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return ([[cond, output]], )
    

class ShowTextCH:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "Achen节点"

    def notify(self, text,unique_id=None, extra_pnginfo=None):        
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:                                    
                    translated_text = [baidutranslationapi(t, 'zh') for t in text]
                    text = translated_text
                    node["widgets_values"] = [text]
            
        return {"ui": {"text": text}, "result": (text,)}
    
class ShowText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "utils"

    def notify(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None:
            if not isinstance(extra_pnginfo, list):
                print("Error: extra_pnginfo is not a list")
            elif (
                not isinstance(extra_pnginfo[0], dict)
                or "workflow" not in extra_pnginfo[0]
            ):
                print("Error: extra_pnginfo[0] is not a dict or missing 'workflow' key")
            else:
                workflow = extra_pnginfo[0]["workflow"]
                node = next(
                    (x for x in workflow["nodes"] if str(x["id"]) == str(unique_id[0])),
                    None,
                )
                if node:
                    node["widgets_values"] = [text]

        return {"ui": {"text": text}, "result": (text,)}

class CollectTextSave:
    @classmethod
    def INPUT_TYPES(s):        
        start_name=read_json_name(read_file_data(start_file))
        return {            
            "optional": {
                "base_text": (start_name,)
            }
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ('text_positive','text_negative',)
    FUNCTION = "exec"
    CATEGORY = "Achen节点"
    OUTPUT_NODE = True

    def exec(self, base_text):        
        base_p,base_n = return_file_value(base_text,start_file,value_a="text_p",value_b="text_n") 
        return (base_p,base_n,)



class TextSplitter: 
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {      
                "text": ("STRING", {"forceInput": True}),          
                "delimiter": ("STRING", {"multiline": False,"default": "English Prompt"}), 
            },            
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "split_text"
    CATEGORY = "Achen Node"

    def split_text(self, text, delimiter):
        # Use regular expression to match Chinese and English titles, and specify re.DOTALL flag
        pattern = re.compile(rf"{delimiter}\n(.*?)$", re.DOTALL)
        
        # Match and extract content
        match = re.search(pattern, text.strip())
        if match:
            english_text = match.group(1).strip()
            return (english_text,)
        else:
            return (text,)


NODE_CLASS_MAPPINGS = {
    "Latent_nodes":Latent_nodes,
    "ClipTextCNEncode": ClipTextCNEncode,
    "CollectTextSave": CollectTextSave,
    #"TextSplitter": TextSplitter,
    #"ShowTextCH": ShowTextCH,
    "ShowText": ShowText,
}


# 包含comfyui节点/可读的标题的字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "Latent_nodes":"空Latent",
    "ClipTextCNEncode": "条件编辑（中文）",
    "CollectTextSave": "提示词收集",
    #"TextSplitter": "分割文本器",
    #"ShowTextCH": "展示文本（中文）",
    "ShowText": "展示文本",
}
