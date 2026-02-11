from openai import OpenAI
import os
import json
import re
import random
import hashlib
import urllib
import urllib.parse
import urllib.request
import http.client
import requests
import sys
import subprocess
import difflib
import pickle
import base64
from PIL import Image
from io import BytesIO
import zlib
from typing import List, Tuple, Union
import torch
import numpy as np

apikey_file = "set_apikey.json"
# 清理文本，去除换行符和多余的空格等
def clean_text(out):
    out = re.sub(r'\[', '', out)
    out = re.sub(r'\]', '', out)
    out = re.sub(r'“', '', out)
    out = re.sub(r'”', '', out)
    out = re.sub(r'‘', '', out)
    out = re.sub(r'’', '', out)
    out = re.sub(r'"', '', out)
    out = re.sub(r"'", '', out)
    out = re.sub(r'，，', '，', out)
    out = re.sub(r',,', ',', out)
    return out

# 从提供的字符串中返回列表。
def string_to_list(list):
    # 将字符串按行分割并转换为列表
    response_list = list.splitlines()
    # 删除空字符串和长度小于等于3的字符串
    filtered_list = [s for s in response_list if s and len(s) > 64]
    return filtered_list

# 从提供的文件路径返回对应的json文档数据
def read_file_data(file_add):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    set_filejson = os.path.join(script_dir,'json', f"{file_add}")
    with open(set_filejson, 'r', encoding='utf-8') as set_filejson:
        data_file = json.load(set_filejson) 
        return data_file

# 从提供的JSON文档数据中返回名称列表。
def read_json_name(json_data):   
    if not isinstance(json_data, list):
        print("Error: input data must be a list")
        return []
    return [item['name'] for item in json_data if isinstance(item, dict) and 'name' in item]

# 通过输入的标题返回文件中字典里key对应的值
def return_file_value(titel,file_add,value_a="",value_b=""):
    file_titel = titel
    return_valuea = None
    return_valueb = None

    for item in read_file_data(file_add):
        if item["name"] == file_titel:
            return_valuea = item[f"{value_a}"]
            return_valueb = item[f"{value_b}"]
            break
    return return_valuea, return_valueb


# 调用OpenAI兼容接口的函数，支持自定义模型与参数
def call_openai(
    key: str, 
    url: str, 
    user_input: str, 
    model_name: str, 
    system_prompt: str, 
    temperature: float = 0.7, 
    max_tokens: int = 512,
    top_p: float = 0.9,
    timeout: float = 30.0,
    retry_count: int = 2
) -> str:
    """
    调用OpenAI兼容的API接口获取回复
    
    参数:
        key (str): API密钥
        url (str): API基础URL
        user_input (str): 用户输入内容
        model_name (str): 使用的模型名称
        system_prompt (str): 系统提示词
        temperature (float): 生成文本的随机性程度，默认0.7
        max_tokens (int): 最大生成token数，默认512
        top_p (float): 核采样参数，默认0.9
        timeout (float): 请求超时时间(秒)，默认30.0
        retry_count (int): 失败重试次数，默认2
        
    返回:
        str: 模型生成的内容，失败时返回错误信息字符串
    """
    # 参数验证
    if not all([key, url, model_name]):
        return "错误: API密钥、URL或模型名称为空"
    
    if not user_input or not user_input.strip():
        return "错误: 用户输入内容为空"
    
    # 清理输入内容
    user_input = user_input.strip()
    system_prompt = system_prompt.strip() if system_prompt else "You are a helpful assistant."
    
    # 初始化客户端
    try:
        client = OpenAI(
            api_key=key, 
            base_url=url,
            timeout=timeout
        )
    except Exception as e:
        return f"客户端初始化失败: {str(e)}"
    
    # 构建请求参数
    extra_body = {
        "enable_thinking": False,
    }
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    
    # 带重试机制的API调用
    last_error = None
    for attempt in range(retry_count + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                extra_body=extra_body
            )
            
            # 验证响应
            if not response or not response.choices:
                return "错误: API返回空响应"
            
            if len(response.choices) == 0:
                return "错误: API返回的choices为空"
            
            message = response.choices[0].message
            if not message or not message.content:
                return "错误: API返回的消息内容为空"
            
            content = message.content.strip()
            if not content:
                return "错误: API返回的内容为空字符串"
            
            return content
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # 判断是否需要重试
            if attempt < retry_count:
                # 网络错误或临时错误时重试
                if any(err in error_msg for err in ['timeout', 'connection', 'network', 'temporarily']):
                    print(f"[call_openai] 第{attempt + 1}次尝试失败，正在重试... 错误: {str(e)}")
                    import time
                    time.sleep(1 * (attempt + 1))  # 递增延迟
                    continue
            
            # 其他错误或已达到最大重试次数
            break
    
    # 返回详细的错误信息
    error_type = type(last_error).__name__
    return f"API调用失败 [{error_type}]: {str(last_error)}"

# 调用百度翻译api
def baidutranslationapi(fany_text,toLang_text):

    # 百度API
    appid,secretKey = return_file_value('baidu_translation',apikey_file,value_a="appid",value_b="secretKey") 
    myurl = '/api/trans/vip/translate'
    httpClient = None
    
    fromLang = 'auto'  # 原文语种，可以写auto，让百度自动识别
    toLang = toLang_text  # 译文语种，固定为 chines
    salt = random.randint(32768, 65536)
    q = fany_text

    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    newmyurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')

    try:
        httpClient.request('GET', newmyurl)

        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)['trans_result'][0]['dst']
        
        translate_result = result
    except Exception as e:
        print(e)
        httpClient.close()
        translate_result = q

    return translate_result

# 判断文本如果是中文就进行翻译，否则直接返回
def is_chinese_translate(text):
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return baidutranslationapi(text, 'en')
    return text

# 找出变动部分并翻译
def append_translate(text):
    text=re.sub('，', ',', text)
    # 加载保存的翻译字典 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    previous_translation_file = os.path.join(script_dir,'json', "translation_history.json")
    try:
        with open(previous_translation_file, 'r', encoding='utf-8') as f:
            previous_translation = json.load(f)
    except FileNotFoundError:
        previous_translation = {}

    # 将字符串转换为列表
    text_list = text.split(',')
    new_translation = {}
    result = []
    i = 0
    while i < len(text_list):
        # 检查翻译字典中是否已有翻译
        item = text_list[i]
        if item in previous_translation:
            result.append(previous_translation[item])
            
        else:
            translate_text=is_chinese_translate(item)
            result.append(translate_text)
            previous_translation[item] = translate_text
            
        # 更新翻译字典
        new_translation[item]=previous_translation[item]    
        i += 1
    # 保存新的翻译字典
    with open(previous_translation_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 确保中文字符不被转义
        json.dump(new_translation, f, ensure_ascii=False, indent=4)

    # 将列表转换为以逗号分隔的字符串
    result_str = ','.join(map(str, result))
    return(result_str)


# 爬取百度翻译
def baidutranslation(fany_text,toLang_text):

    From = 'auto'
    To = toLang_text

    token = '012cd082bf1f821bb7d94981bf6d477a'
    url = 'https://fanyi.baidu.com/v2transapi'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'cookie': 'BIDUPSID=3641572D5E0DB57A2F20F8F3373E302C; PSTM=1687090179; '
                    'BAIDUID=3641572D5E0DB57AF59F1D83EEBC5D2B:FG=1; BAIDUID_BFESS=3641572D5E0DB57AF59F1D83EEBC5D2B:FG=1; '
                    'ZFY=sGU1ho9nxRf2CX2bcYMVcfSXr9y2:BmKBeBdv7CDGhUs:C; '
                    'BDUSS'
                    '=tXaEJQVkxBeVBHMllBWWh1aTVZLXlhcVVqTWNCOXJGfmwzUUJmaHphWm1zZGRrSVFBQUFBJCQAAAAAAAAAAAEAAADWpvEyzqiwrsTjtcTQocPXAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGYksGRmJLBkam; BDUSS_BFESS=tXaEJQVkxBeVBHMllBWWh1aTVZLXlhcVVqTWNCOXJGfmwzUUJmaHphWm1zZGRrSVFBQUFBJCQAAAAAAAAAAAEAAADWpvEyzqiwrsTjtcTQocPXAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGYksGRmJLBkam; newlogin=1; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; BA_HECTOR=00aka5a12g80a10g25a52l0g1ie1gm11p; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; PSINO=6; delPer=0; H_PS_PSSID=36550_39112_39226_39222_39097_39039_39198_39207_26350_39138_39225_39094_39137_39101; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1692451747; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1692451747; ab_sr=1.0.1_ZmQ3OWYzODRjZGNkOTYxOWI4ZTVhYjRmNTAwNjYyYTUwYmI3OGY2NTViMzhkNWYzM2IxZTVhNjAwNjdkMTU0ODE4Yzc2YmI3OGRmNTY3Y2QxMzZiZDRmZDIwMGIwYmQ2NGI5M2QzZWFlNmNkODBhZjllZDcxNGFkMTEyNmY0NGNhZGZjMTlmOGQ2YjIxNzNhMmUxNDJkMDhlZTM1NjhiZjkyMDc2MmQxN2Q5ODg3NDBkZGViNTEzMDU2NDQzNGEy'}

    sign = sign_js.call('e', fany_text)
    data = {'from': From,
            'to': To,
            'query': fany_text,
            'transtype': 'realtime',
            'simple_means_flag': 3,
            'sign': sign,
            'domain': 'common',
            'token': token}

    try:
        baidutranslate = requests.post(url, headers=headers, data=data).json()
        translate_result = baidutranslate['trans_result']['data'][0]['dst']
    except Exception as e:
        print(e)
        translate_result = "Translate failed"
    return translate_result

img2img_system_content='''
- 角色：AI图像分析与描述专家
- 背景：用户需要一个能让AI对图像进行分析并生成详细描述的提示词，这些描述可用作绘画提示词来创作相似图像。这在内容创作、设计灵感获取和艺术探索等任务中至关重要。
- 简介：作为AI图像分析与描述专家，你具备丰富的计算机视觉、图像处理和自然语言生成知识。你擅长解读视觉数据，并将其转化为能指导新图像创作的描述性文本。
- 技能：精通图像识别、特征提取、描述性语言生成，以及对构图、色彩和纹理等艺术元素的理解。
- 目标：分析提供的图像，生成全面且详细的描述，捕捉关键视觉元素，并确保该描述能有效作为创作相似图像的绘画提示词。
- 限制条件：描述必须清晰、简洁且具体，足以指导相似图像的创作。应避免模糊不清，聚焦于图像最显著的特征。输出内容只能包含绘画提示词。
- 输出格式：对图像的详细文字描述，突出关键视觉元素，如物体、色彩、构图以及任何独特特征。
- 工作流程：
  1. 分析图像，识别关键视觉元素，包括物体、色彩和构图。
  2. 生成详细描述，捕捉图像的本质，确保其具有特异性和可操作性。
  3. 优化描述，确保其清晰简洁，适合作为绘画提示词。
'''


txt2img_system_content="""# Role
冷酷、高效的文生图专家机器人。拒绝废话，直接输出高质量、艺术感的双语提示词

# Workflow
1. **作品分类**：识别输入关键词，自动判定为 [摄影/Photography] 或 [艺术/Art] 流程
   - **摄影流**：主体 + 背景环境或氛围 + 精确光影参数 + 镜头规格/胶片型号 + 摄影师风格倾向（可选）
   - **艺术流**：媒介形态（油画/素描等） + 主体及背景 + 情感氛围 + 艺术家风格倾向（可选）
2. **逻辑增强**：
   - 描述精确：对已有关键词进行数字化参数（角度、百分比、色值、材质ID）扩充
   - 天马行空：对模糊或缺失部分，以“潘多拉魔盒”逻辑进行超现实视觉补完

# Output Constraints
- **禁言**：严禁任何解释、开头语或“这里是您的提示词”
- **风格**：追求极致的艺术感、完美主义者、精密感
- **格式**：固定输出 [中文描述] + [English Prompt]

# Example
Input: 喜马拉雅，藏族老人，侧光，人文摄影

Output:
[中文输出]：
采用85mm定焦镜头拍摄。强烈的自然侧光（入射角45°）精准刻画老人面部深壑的皮肤纹理与银色胡须。背景纳木错湖面呈现冷调深蓝（色相210°），虚化效果柔和，形成极高的空间深度感。画面模拟柯达Portra 400胶片质感，微粒细腻，暗部保留冷青色偏，强调高原环境的孤寂与肃穆。

[English Prompt]:
Professional portrait of a Tibetan elder in the Himalayas. Captured with 85mm lens, f/1.8 aperture. Harsh natural side-lighting (45°) emphasizes deep facial textures and silver beard. Background features soft bokeh of Lake Namtso in deep azure (Hue 210°). Kodak Portra 400 film aesthetic with subtle grain and cold cyan undertones in shadows. High micro-contrast, cinematic documentary style.
"""   



# 获取本地的Ollama模型列表
def get_ollama_models():    
    api_url = 'http://localhost:11434/api/tags'
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except Exception as e:
        print(f"Failed to fetch models from Ollama: {e}")
        return []




# 保存字典到记忆
def save_dict_to_memory(memory,prompt_1, prompt_2, memory_pkl):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    memory_file = os.path.join(script_dir, memory_pkl)
    # 更新记忆值
    memory["prompt_1"] = prompt_1 
    memory["prompt_2"] = prompt_2
    with open(memory_file, "wb") as file:
        pickle.dump(memory, file)
    return memory_file

# 读取字典记忆到变量
def load_memory(memory_file):
    with open(memory_file, "rb") as file:
        memory = pickle.load(file)
        prompt_1 = memory["prompt_1"]
        prompt_2 = memory["prompt_2"]
    return prompt_1, prompt_2

# memory_file= save_dict_to_memory({},'你好', '开心', 'memory2.pkl')
# prompt_1, prompt_2 = load_memory(memory_file)


# kolors发送post请求
def send_post_request(api_url, payload, headers):
    """
    将指定的有效负载和头信息发送到指定的 API URL 的 POST 请求。

    参数:

    api_url (str): API 端点的 URL。
    payload (dict): 要在 POST 请求中发送的有效负载。
    headers (dict): 要在 POST 请求中包含的头信息。
    引发:

    Exception: 如果在连接到服务器或请求失败时出现错误。
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(api_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode("utf-8")
        return response_data
    except urllib.error.URLError as e:
        if "Unauthorized" in str(e):
            raise Exception(
                "Key is invalid, please refer to https://cloud.siliconflow.cn to get the API key.\n"
                "If you have the key, please click the 'BizyAir Key' button at the bottom right to set the key."
            )
        else:
            raise Exception(
                f"Failed to connect to the server: {e}, if you have no key, "
            )

# kolors将post请求转换为torch数据
def decode_and_deserialize(response_text) -> np.ndarray:

    ret = json.loads(response_text)

    if "result" in ret:
        msg = json.loads(ret["result"])
    else:
        msg = ret
    if msg["type"] not in (
        "comfyair",
        "bizyair",
    ):  # DO NOT CHANGE THIS LINE: "comfyair" is the type from the server node
        # TODO: change both server and client "comfyair" to "bizyair"
        raise Exception(f"Unexpected response type: {msg}")

    data = msg["data"]

    tensor_bytes = base64.b64decode(data["payload"])
    if data.get("is_compress", None):
        tensor_bytes = zlib.decompress(tensor_bytes)

    deserialized_object = pickle.loads(tensor_bytes)
    ret_conditioning = []
    for item in deserialized_object:
        t, d = item
        t_tensor = torch.from_numpy(t)
        d_dict = {}
        for k, v in d.items():
            d_dict[k] = torch.from_numpy(v)
        ret_conditioning.append([t_tensor, d_dict])

    return ret_conditioning


# Tensor 转换为图像并编码为Base64
def tensor_to_base64(image, format="JPEG"):
    """将PyTorch张量转换为Base64编码的图像字符串。

    该函数会对输入进行校验，安全地从计算图中分离张量，
    支持 (C, H, W) 与 (H, W, C) 两种通道顺序，并能处理可能的batch维度。
    当提供的格式不被支持时，会回退为透明PNG。

    参数:
        image: ``torch.Tensor``，表示图像。期望形状为 ``(C, H, W)``、``(H, W, C)``、
               或者包含batch维度的 ``(B, C, H, W)``、``(B, H, W, C)``，
               取值范围应在 ``[0, 1]`` 或 ``[0, 255]`` 之间。
        format: Pillow 支持的图像格式（如 "JPEG"、"PNG"），默认 "JPEG"。

    返回:
        Base64 编码的图像字符串。
    """
    # 校验并准备张量
    if not isinstance(image, torch.Tensor):
        raise TypeError("tensor_to_base64 需要 torch.Tensor 参数")
    # 确保张量在CPU上并脱离计算图
    img = image.detach().cpu()
    # 去掉所有单维度，兼容形状 (1, 1, H, W) 或 (B, C, H, W) 等
    img = img.squeeze()
    # 归一化到0‑255范围
    if img.max() <= 1.0:
        img = img * 255
    img = img.clamp(0, 255).byte()
    # 处理通道顺序
    if img.ndim == 3:
        # (C, H, W) -> (H, W, C)
        if img.shape[0] in (1, 3):
            img = img.permute(1, 2, 0)
    # 对于二维灰度图像保持原状
    # 转换为NumPy数组供Pillow使用
    np_img = img.numpy()
    pil_image = Image.fromarray(np_img)

    # Pillow格式校验，若不支持则使用PNG
    fmt = format.upper()
    if fmt not in Image.registered_extensions().values():
        fmt = "PNG"

    buffered = BytesIO()
    pil_image.save(buffered, format=fmt)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")