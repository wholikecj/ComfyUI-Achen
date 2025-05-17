from openai import OpenAI
import os
import base64
import mimetypes
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import json

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
    clean_text,)


apikey_file = "set_apikey.json"   

class Txt2ImgContent:  
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        apikey_name=read_json_name(read_file_data(apikey_file))
        return {
            "required": {
                "model_name": (apikey_name,), 
                "temperature_value": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}), 
                "prompt": ("STRING", {"multiline": True,"default": "赛博朋克!，冷白光>9500K，女工程师@NeoTokyo_Ver12，身体前倾>18°，悬浮车@HoverPod_MK3"}),  
                # "end": ("BOOLEAN", {"default": False},), 
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("CH","EN",)
    FUNCTION = "Api_anything"    
    OUTPUT_NODE = True
    
    CATEGORY = "Achen节点"

    def Api_anything(self,prompt,model_name,temperature_value):
        
       # 判断是否调用OpenAI函数扩写句子
        if model_name == "None":
            prompt_output_ch=prompt
            prompt_output_en=append_translate(prompt)
        else:
            system = txt2img_system_content
            url,key = return_file_value(model_name,apikey_file,value_a="url",value_b="key") 
            response_date=call_openai(key,url,prompt,model_name,system,temperature_value)# 调用openai

            # 调用函数将字符串转换为列表，并处理空行和小于64个字符的行
            filtered_list=string_to_list(response_date)
            prompt_output_ch=filtered_list[0]
            # 检查列表长度是否大于1，然后使用索引1获取第二个元素，否则使用翻译函数进行翻译
            prompt_output_en = filtered_list[1] if len(filtered_list) > 1 else baidutranslationapi(prompt_output_ch) 
        return (prompt_output_ch,prompt_output_en)

class Txt2ImgContentSystem:  
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        apikey_name=read_json_name(read_file_data(apikey_file))
        system_default = txt2img_system_content
        return {
            "required": {
                "model_name": (apikey_name,), 
                "temperature_value": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.1}), 
                "prompt": ("STRING", {"multiline": True,"default": "赛博朋克!，冷白光>9500K，女工程师@NeoTokyo_Ver12，身体前倾>18°，悬浮车@HoverPod_MK3"}),
                "system": ("STRING", {"multiline": True,"default": f"{system_default}"}),
                # "end": ("BOOLEAN", {"default": False},), 
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("CH","EN",)
    FUNCTION = "Api_anything"    
    OUTPUT_NODE = True
    
    CATEGORY = "Achen节点"

    def Api_anything(self,prompt,system,model_name,temperature_value):
        
       # 判断是否调用OpenAI函数扩写句子
        if model_name == "None":
            prompt_output_ch=prompt
            prompt_output_en=append_translate(prompt)
        else:            
            url,key = return_file_value(model_name,apikey_file,value_a="url",value_b="key") 
            response_date=call_openai(key,url,prompt,model_name,system,temperature_value)# 调用openai

            # 调用函数将字符串转换为列表，并处理空行和小于64个字符的行
            filtered_list=string_to_list(response_date)
            prompt_output_ch=filtered_list[0]
            # 检查列表长度是否大于1，然后使用索引1获取第二个元素，否则使用翻译函数进行翻译
            prompt_output_en = filtered_list[1] if len(filtered_list) > 1 else baidutranslationapi(prompt_output_ch) 
        return (prompt_output_ch,prompt_output_en)

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

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def to_base64(image, ):
      import base64
      from io import BytesIO

      # 将张量图像转换为PIL图像
      pil_image = tensor2pil(image)

      buffered = BytesIO()
      pil_image.save(buffered, format="JPEG")
      image_bytes = buffered.getvalue()

      base64_str = base64.b64encode(image_bytes).decode("utf-8")
      return base64_str 

class Api_ImageAnything:  
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        apikey_name=read_json_name(read_file_data(apikey_file))
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (apikey_name,), 
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_image_response"
    CATEGORY = "Achen节点"

    def get_image_response(self, model_name,image): 
        system = img2img_system_content
        url,key = return_file_value(model_name,apikey_file,value_a="url",value_b="key") 

        client = OpenAI(api_key=key, base_url=url)

        mime_type= "image/png"

        encoded_image_str = to_base64(image, )
        # 创建数据前缀
        data_uri_prefix = f'data:{mime_type};base64,'
        # 拼接前缀和Base64编码的图像数据
        encoded_image_str = data_uri_prefix + encoded_image_str
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": system
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image_str
                            }
                        }
                    ]
                }
            ],
            top_p=0.8,
            stream=True,
            stream_options={"include_usage": True}
        )
        response_list = "\n".join([chunk.model_dump_json() for chunk in completion])
        # 将字符串拆分成多个 JSON 对象
        json_chunks = response_list.strip().split('\n')

        # 提取中文内容
        chinese_content = ""
        for chunk in json_chunks:
            try:
                json_data = json.loads(chunk)
                if 'choices' in json_data and len(json_data['choices']) > 0:
                    delta = json_data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    chinese_content += content
            except json.JSONDecodeError:
                continue

        return(chinese_content,)
   


NODE_CLASS_MAPPINGS = {
    "Txt2ImgContent":Txt2ImgContent,
    "Txt2ImgContentSystem":Txt2ImgContentSystem,
    "Api_ImageAnything":Api_ImageAnything,
    "KolorsTextEncode":KolorsTextEncode,
}


# 一个包含节点友好/可读的标题的字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "Txt2ImgContent":"文本生图（AI）", 
    "Txt2ImgContentSystem": "文本生图（系统）",
    "Api_ImageAnything":"图像反推（AI）", 
    "KolorsTextEncode":"条件编辑（Kolors）",
}