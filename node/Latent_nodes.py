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

class LatentPresetSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ratio": (["1:1", "3:2", "4:3", "5:4", "16:9", "16:10"], {"default": "3:2"}),
                "long_edge": ([512, 768, 1024, 1280, 1536, 1920, 2048], {"default": 1024}),
                "landscape": ("BOOLEAN", {"default": False, "label_on": "横向", "label_off": "竖向"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Achen节点"

    def generate(self, ratio, long_edge, landscape, batch_size):
        # 解析比例 (宽:高)
        ratio_map = {
            "1:1": (1, 1),
            "3:2": (3, 2),
            "4:3": (4, 3),
            "5:4": (5, 4),
            "16:9": (16, 9),
            "16:10": (16, 10),
        }
        w_ratio, h_ratio = ratio_map[ratio]

        # 长边始终是 long_edge
        if landscape:
            # 横向：宽是长边，高是短边
            width = long_edge
            height = int(width * h_ratio / w_ratio)
        else:
            # 竖向：高是长边，宽是短边
            height = long_edge
            width = int(height * h_ratio / w_ratio)

        # 确保都是8的倍数
        width = (width // 8) * 8
        height = (height // 8) * 8

        # VAE latent 尺寸是图像尺寸除以 8
        latent_width = width // 8
        latent_height = height // 8

        # 创建空白 latent（全零）
        latent = torch.zeros([batch_size, 4, latent_height, latent_width])

        return ({"samples": latent},)
    
class ClipTextCNEncode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        types = {
            "required": {
                "CLIP": ("CLIP", {"tooltip": "用于对文本进行编码的CLIP模型。"}),
                "text": ("STRING", {"forceInput": True}), 
            },
        }
        return types

    RETURN_TYPES = ('CONDITIONING',)
    FUNCTION = 'prompt_text'
    CATEGORY = 'Achen节点'

    def prompt_text(self,CLIP,text):          
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
        try:
            start_name = read_json_name(read_file_data(start_file))
        except:
            start_name = []
        # 在列表开头添加保存模式选项
        start_name.insert(0, "【保存新提示词】")
        # 确保列表不为空（至少有保存选项）
        if len(start_name) == 1:
            start_name.append("（暂无保存的提示词）")
        return {
            "required": {
                "base_text": (start_name, {"default": start_name[0]}),
                "new_text_p": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ('positive',)
    FUNCTION = "exec"
    CATEGORY = "Achen节点"
    OUTPUT_NODE = True

    def exec(self, base_text="", new_text_p=""):
        # 根据下拉列表选择判断模式：第一条为保存模式，其他为读取模式
        if base_text == "【保存新提示词】":
            # 保存模式
            if not new_text_p.strip():
                print("[CollectTextSave] 保存模式：请输入要保存的提示词")
                return ("",)
            
            try:
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                json_dir = os.path.join(script_dir, 'json')
                json_path = os.path.join(json_dir, start_file)
                
                # 如果目录不存在则创建
                if not os.path.exists(json_dir):
                    os.makedirs(json_dir)
                
                # 读取现有数据
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = []
                
                # 获取基础标题（截取第一个逗号前的内容，无逗号则取前30字符）
                import re
                text_content = new_text_p.strip()
                # 找到第一个中英文句号的位置
                comma_match = re.search(r'[.。]', text_content)
                if comma_match:
                    # 有句号，截取句号前的内容
                    base_title = text_content[:comma_match.start()].strip()[:100]
                else:
                    # 无句号，取前30字符
                    base_title = text_content[:30]
                
                # 判断是否为英文（包含英文字母）
                if re.search(r'[a-zA-Z]', base_title):
                    # 是英文，调用百度翻译API翻译成中文
                    try:
                        translated_title = baidutranslationapi(base_title, 'zh')
                        if translated_title and translated_title != base_title:
                            base_title = translated_title
                    except Exception as trans_err:
                        # 翻译失败则保持原标题
                        pass
                
                new_title = base_title  # 默认标题
                
                # 检查是否已存在相同基础标题，存在则覆盖
                is_update = False
                for item in data:
                    # 移除序号进行比较
                    existing_name = item.get("name", "")
                    if "." in existing_name:
                        existing_base = existing_name.split(".", 1)[1] if "." in existing_name else existing_name
                    else:
                        existing_base = existing_name
                    
                    if existing_base == base_title:
                        item["text_p"] = new_text_p
                        is_update = True
                        new_title = existing_name  # 保持原有带序号的标题
                        break
                
                # 如果是新标题则添加序号
                if not is_update:
                    # 生成序号（当前条目数+1）
                    index = len(data) + 1
                    new_title = f"{index}.{base_title}"
                    new_entry = {
                        "name": new_title,
                        "text_p": new_text_p
                    }
                    data.append(new_entry)
                
                # 保存到文件
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                action = "更新" if is_update else "保存"
                print(f"[CollectTextSave] 成功{action}: {new_title}")
                return (new_text_p,)
            except Exception as e:
                print(f"[CollectTextSave] 保存失败: {str(e)}")
                return (new_text_p,)
        
        else:
            # 读取模式
            if not base_text or base_text in ["（暂无保存的提示词）"]:
                print("[CollectTextSave] 读取模式：没有可读取的提示词")
                return ("",)
            
            try:
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                json_path = os.path.join(script_dir, 'json', start_file)
                
                # 读取数据
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 查找对应标题的提示词
                base_p = ""
                for item in data:
                    if item.get("name") == base_text:
                        base_p = item.get("text_p", "")
                        break
                
                print(f"[CollectTextSave] 已读取: {base_text}")
                return (base_p,)
            except Exception as e:
                print(f"[CollectTextSave] 读取失败: {str(e)}")
                return ("",)

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
    "LatentPresetSize":LatentPresetSize,
    "ClipTextCNEncode": ClipTextCNEncode,
    "CollectTextSave": CollectTextSave,
    #"TextSplitter": TextSplitter,
    #"ShowTextCH": ShowTextCH,
    "ShowText": ShowText,
}


# 包含comfyui节点/可读的标题的字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentPresetSize":"Latent(空)",
    "ClipTextCNEncode": "条件编辑（输入）",
    "CollectTextSave": "提示词收集",
    #"TextSplitter": "分割文本器",
    #"ShowTextCH": "展示文本（中文）",
    "ShowText": "展示文本",
}
