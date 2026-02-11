from openai import OpenAI
import os
import base64
import mimetypes
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import json
import requests
import time
import torch
from torchvision import transforms

from ..tools import (
    read_file_data,
    read_json_name,
    return_file_value,
    call_openai,
    append_translate,
    txt2img_system_content,
    img2img_system_content,
    send_post_request,
    decode_and_deserialize,
    baidutranslationapi,
    string_to_list,
    clean_text,
    tensor_to_base64    
)


apikey_file = "set_apikey.json"   

class Txt2ImgContent:  
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        apikey_name = read_json_name(read_file_data(apikey_file))
        return {
            "required": {
                "model_name": (apikey_name,), 
                "temperature_value": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}), 
                "prompt": ("STRING", {"multiline": True, "default": "宫崎骏电影画面，龙猫，村庄，大树"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("CH", "EN")
    FUNCTION = "Api_anything"    
    OUTPUT_NODE = True
    CATEGORY = "Achen节点"

    def Api_anything(self, prompt, model_name, temperature_value):
        if isinstance(prompt, list):
            prompt = ", ".join(str(item) for item in prompt)
        
        if not prompt or not prompt.strip():
            return ("", "")
        
        if model_name == "None":
            prompt_output_ch = prompt
            prompt_output_en = append_translate(prompt)
        else:
            try:
                url, key = return_file_value(model_name, apikey_file, value_a="url", value_b="key")
                if not url or not key:
                    return (prompt, append_translate(prompt))
                response_date = call_openai(key, url, prompt, model_name, txt2img_system_content, temperature_value)
                filtered_list = string_to_list(response_date)
                prompt_output_ch = filtered_list[0] if filtered_list else prompt
                prompt_output_en = filtered_list[1] if len(filtered_list) > 1 else baidutranslationapi(prompt_output_ch)
            except Exception as e:
                print(f"[Txt2ImgContent] API调用错误: {e}")
                prompt_output_ch = prompt
                prompt_output_en = append_translate(prompt)
        
        return (prompt_output_ch, prompt_output_en)

class Txt2ImgContentSystem:  
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        apikey_name = read_json_name(read_file_data(apikey_file))
        return {
            "required": {
                "model_name": (apikey_name,), 
                "temperature_value": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}), 
                "prompt": ("STRING", {"forceInput": True}),
                "system": ("STRING", {"multiline": True, "default": txt2img_system_content}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("CH", "EN")
    FUNCTION = "Api_anything"    
    OUTPUT_NODE = True
    CATEGORY = "Achen节点"

    def Api_anything(self, prompt, system, model_name, temperature_value):
        if isinstance(prompt, list):
            prompt = ", ".join(str(item) for item in prompt)
        
        if not prompt or not prompt.strip():
            return ("", "")
        
        if model_name == "None":
            prompt_output_ch = prompt
            prompt_output_en = append_translate(prompt)
        else:
            try:
                url, key = return_file_value(model_name, apikey_file, value_a="url", value_b="key")
                if not url or not key:
                    return (prompt, prompt)
                response_date = call_openai(key, url, prompt, model_name, system, temperature_value)
                filtered_list = string_to_list(response_date)
                prompt_output_ch = filtered_list[0] if filtered_list else prompt
                prompt_output_en = filtered_list[1] if len(filtered_list) > 1 else baidutranslationapi(prompt_output_ch)
            except Exception as e:
                print(f"[Txt2ImgContentSystem] API调用错误: {e}")
                prompt_output_ch = prompt
                prompt_output_en = append_translate(prompt)
        
        return (prompt_output_ch, prompt_output_en)

class KolorsTextEncode: 
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "kolrosencode"
    CATEGORY = "Achen节点"

    def kolrosencode(self, text):
        url,key = return_file_value('None',apikey_file,value_a="url",value_b="key") 
        url = "https://bizyair-api.siliconflow.cn/x/v1/supernode/mzkolorschatglm3"

        payload = {
            "text": text,
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        response: str = send_post_request(url, payload=payload, headers=headers)
        tensors_np = decode_and_deserialize(response)

        return (tensors_np,)


class Api_ImageAnything:
    """
    图像识别节点：调用OpenAI API进行图像分析和描述
    
    功能：
    - 上传图像并获取AI的描述和分析
    - 支持自定义提示词和系统角色
    - 可调节temperature等参数控制生成质量
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        apikey_name = read_json_name(read_file_data(apikey_file))
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (apikey_name,), 
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "先描述什么风格，再详细描述画面内容", 
                    "multiline": True,
                    "tooltip": "用户输入的提示词，用于指导AI如何分析图片"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "温度参数，值越高输出越随机，越低越确定"
                }),
                "max_tokens": ("INT", {
                    "default": 512, 
                    "min": 256, 
                    "max": 1024, 
                    "step": 64,
                    "tooltip": "最大生成token数"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_image_response"
    CATEGORY = "Achen节点"

    def get_image_response(self, model_name, image, prompt="先描述风格，再详细描述画面内容", 
                          system_prompt=img2img_system_content,
                          temperature=0.7, max_tokens=512, top_p=0.8):
        """
        调用OpenAI API进行图像识别
        
        参数:
            model_name: 模型名称
            image: 输入图像张量
            prompt: 用户提示词
            system_prompt: 系统角色设定
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样参数
            
        返回:
            tuple: 包含识别结果字符串
        """
        try:
            # 获取API配置
            url, key = return_file_value(model_name, apikey_file, value_a="url", value_b="key")
            
            if not key or not url:
                return ("错误：未找到有效的API密钥或URL配置",)
            
            # 初始化OpenAI客户端
            client = OpenAI(api_key=key, base_url=url, timeout=60.0)
            
            # 转换图像为base64编码
            encoded_image_str = tensor_to_base64(image)
            if not encoded_image_str:
                return ("错误：图像转换失败",)
            
            # 创建数据URI（使用JPEG格式）
            data_uri = f'data:image/jpeg;base64,{encoded_image_str}'
            
            # 构建消息列表（分离system和user角色）
            messages = [
                {
                    "role": "system",
                    "content": system_prompt if system_prompt else "你是一位专业的图像分析助手。"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }
            ]
            
            # 调用API（非流式输出）
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False
            )
            
            # 处理响应
            if not response or not response.choices:
                return ("错误：API返回空响应",)
            
            message = response.choices[0].message
            if not message or not message.content:
                return ("错误：API返回的内容为空",)
            
            result_text = message.content.strip()
            if not result_text:
                return ("警告：未获取到有效响应内容",)
                
            return (result_text,)
            
        except Exception as e:
            error_msg = f"API调用错误: {str(e)}"
            print(f"[Api_ImageAnything] {error_msg}")
            return (error_msg,)
   
class Api2ImgNods:
    # 默认超时时间（秒）
    timeout = 120
    # 最大轮询总时间（秒）
    max_poll_time = 90

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"multiline": False, "default": "MAILAND/majicflus_v1"}),
                "api_key": ("STRING", {"multiline": False, "default": "ms-"}),
                "width": ("INT", {"default": 760, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 999999, "step": 1}),
                "prompt": ("STRING", {"multiline": True, "default": "赛博朋克!，冷白光>9500K，女工程师@NeoTokyo_Ver12，身体前倾>18°，悬浮车@HoverPod_MK3"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "Api_t2i"
    OUTPUT_NODE = True
    CATEGORY = "Achen节点"

    def Api_t2i(self, prompt, model, api_key, width, height, seed):
        """通过ModelScope API生成图像，返回 (image_tensor,) 或错误信息。"""
        start_time = time.time()

        # 参数校验
        if not (512 <= width <= 2048 and 512 <= height <= 2048):
            return ("错误：宽高超出范围",)
        if seed < 0:
            return ("错误：seed 必须为非负整数",)

        base_url = 'https://api-inference.modelscope.cn/'
        common_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # 发送生成请求
        try:
            t0 = time.time()
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps({
                    "model": model,
                    "prompt": prompt,
                    "size": f"{width}x{height}",
                    "seed": seed,
                }, ensure_ascii=False).encode('utf-8'),
                timeout=self.timeout
            )
            response.raise_for_status()
            task_id = response.json().get("task_id")
            if not task_id:
                return ("错误：未获取 task_id",)
            print(f"[Api2ImgNods] 提交任务耗时: {time.time() - t0:.2f}s, task_id: {task_id}")
        except Exception as e:
            print(f"[Api2ImgNods] 请求失败: {e}")
            return (f"API请求错误: {e}",)

        # 动态轮询：初始间隔0.5秒，逐步增加到最大2秒
        poll_interval = 0.5
        max_interval = 2.0

        while time.time() - start_time < self.max_poll_time:
            try:
                result_resp = requests.get(
                    f"{base_url}v1/tasks/{task_id}",
                    headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                    timeout=30
                )
                result_resp.raise_for_status()
                data = result_resp.json()
            except Exception as e:
                print(f"[Api2ImgNods] 轮询错误: {e}")
                return (f"轮询错误: {e}",)

            status = data.get("task_status")
            elapsed = time.time() - start_time

            if status == "SUCCEED":
                print(f"[Api2ImgNods] 生成成功，总耗时: {elapsed:.2f}s")
                img_url = data.get("output_images", [])[0]
                if not img_url:
                    return ("错误：未返回图片 URL",)
                try:
                    img_resp = requests.get(img_url, stream=True, timeout=30)
                    img_resp.raise_for_status()
                    image = Image.open(BytesIO(img_resp.content))
                except Exception as e:
                    return (f"下载图片失败: {e}",)

                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_array = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_array).unsqueeze(0)
                return (image_tensor,)
            elif status == "FAILED":
                return ("错误：生成失败",)

            # 等待后继续轮询，间隔逐步增加
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, max_interval)

        print(f"[Api2ImgNods] 轮询超时，总耗时: {time.time() - start_time:.2f}s")
        return ("错误：轮询超时",)


NODE_CLASS_MAPPINGS = {
    "Txt2ImgContent":Txt2ImgContent,
    "Txt2ImgContentSystem":Txt2ImgContentSystem,
    "Api_ImageAnything":Api_ImageAnything,
    "KolorsTextEncode":KolorsTextEncode,
    "Api2ImgNods":Api2ImgNods,
}


# 一个包含节点友好/可读的标题的字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "Txt2ImgContent":"提示词（AI）", 
    "Txt2ImgContentSystem": "提示词（系统）",
    "Api_ImageAnything":"图像反推（AI）", 
    "KolorsTextEncode":"条件编辑（Kolors）",
    "Api2ImgNods":"图像生成（api）",
}