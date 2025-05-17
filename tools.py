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
try:
    import execjs
except ImportError:
    subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'install', 'pyexecjs'], shell=True)
    import execjs

import base64
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

# 调用openai api
def call_openai(key,url,User_input,model_name,system,temperature_value):
    client = OpenAI(api_key=key, base_url=url)
    response = client.chat.completions.create(
        model=f"{model_name}",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": User_input},
        ],
        max_tokens=512,
        temperature=temperature_value,
        stream=False,  # 非流式调用
        enable_thinking=False  # 必须设置为false
    )
    chat_content = response.choices[0].message.content
    return chat_content

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



# 读取百度翻译的加盐算法
script_dir = os.path.dirname(os.path.abspath(__file__))
js_file_path = os.path.join(script_dir, 'js', 'BaiduTranslate_sign.js')
with open(js_file_path, 'r', encoding='utf-8') as f:
    sign_js = execjs.compile(f.read())

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
    # sign = execjs.compile(js).call("e", q)
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
视觉描述系统架构（output in English）：

#视觉特征结构化框架

##媒体类型识别
类型精准定位：照片（写实/手机摄影/专业单反）/插画（二次元/矢量风/手绘水彩）/3D渲染（赛博朋克/低多边形）
风格特征提取：胶片颗粒感/8bit像素风/霓虹光效/水墨晕染

##物理空间建模
光照体系：强侧光/柔光箱布光/双点光源/自然光黄金时刻
色彩编码：主色调(CMYK值)+辅助色+点缀色占比
透视结构：广角畸变/平行透视/俯视45°视角

##主体要素解构
人物特征：
• 生理特征：东亚女性/卷发高光/小麦肤色
• 动态建模：左手扶腰站立/右腿前交叉
• 服饰语义：oversize牛仔夹克/做旧金属拉链

物体识别：
• 概念组合：蒸汽朋克机械臂/发光苔藓生态瓶
• 材质标注：阳极氧化铝表面/亚克力荧光管

##环境语义网络
空间锚点：咖啡厅吧台/赛博城市天际线/苔原冻土层
氛围元素：丁达尔光线/高斯模糊背景/数字雨特效

#系统自动生成结果用英文输出（output in English）
This is a professionally photographed image using rule-of-thirds composition. Centered in the frame stands an Asian woman on a neon-lit Tokyo street, wearing a matte leather trench coat. She holds a transparent umbrella in her right hand, its surface refracting blue-purple light spots. Blurred billboards in the background display Chinese characters reading "Shinjuku", with warm yellow convenience store lights reflecting on the wet asphalt.

#限制条件：
准确描述：描述应该充分参考原图，尽可能的准确充分的描述清楚，而不是简单的概括性陈述。
输出格式：你给出的图片描述应只包含描述部分，不应出现系统提示的结构，并以英文输出（output in English）。
'''


txt2img_system_content="""智能文生图提示词生成系统 3.0
（输入简单关键词 → 输出专业级描述）

系统核心架构
🛠 五维解析引擎
风格解构模块：赛博朋克 → 霓虹故障特效/生物机械元素/垂直城市结构
人物建模器：女工程师 → 碳纤维外骨骼(反射率≤15%) + 神经交互接口(坐标X:0.55,Y:0.7)
动作翻译器：身体前倾 → 脊柱倾角12° + 手掌悬浮触控态(距离全息屏Δ=0.3m)
光影计算云：冷白光 → 色温9000K + 环境光遮蔽强度0.7
生态生成器：自动补全实验室/悬浮载具/数据雨等场景元素

用户输入示例
🔖 极简指令：
赛博朋克，冷白光，女工程师，身体前倾姿势

系统自动生成结果
中文描述
超现实赛博朋克概念图，采用强对比冷调光源（色温9000K）。女工程师（坐标X:0.5,Y:0.6）身着哑光碳纤维战术服（反射率12%），以12°前倾姿势操作全息控制台（覆盖画面35%）。AR目镜投射蓝色数据流（HEX:#00BFFF），背景的生化实验室玻璃幕墙（厚度=0.8m）外，三架悬浮运输舱正穿透霓虹雨幕（雨滴密度=120/㎡）。地板的反光金属网格（材质ID:CyberGrid_09）延伸出无限纵深透视。

English Prompt
Cyberpunk hyperrealism with high-contrast cold lighting (9000K). A female engineer in matte carbon-fiber suit (reflectivity=12%) leans forward 12° to manipulate holographic panels (35% coverage). AR visor projects azure data streams (HEX:#00BFFF) while three hover pods pierce through neon rainstorm (density=120/m²) beyond bioreactor lab's glass walls (thickness=0.8m). Reflective floor grids (materialID:CyberGrid_09) create endless depth in fisheye perspective.
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
