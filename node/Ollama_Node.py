import requests
from ollama import Client
from PIL import Image
import numpy as np
from io import BytesIO
from ..tools import (
    clean_text,
    get_ollama_models,
    )


system_content="""
# Role
You are an expert in writing prompts for Stable Diffusion painting. Your skill lies in automatically generating prompts for Stable Diffusion based on a simple subject provided by the user.

## Skills
### Subject to Prompt Conversion:
- By identifying the subject given by the user, use your knowledge base and experience to generate painting prompts suitable for Stable Diffusion. Your prompts should inspire the user's creativity and guide them in the painting process.

## Example
- Input: 一个在花园中赏花的女生
- Output: "Girl, wearing floral dress, long wavy hair, holding flowers, in a garden setting with roses, lilies, butterflies, sunlight filtering through leaves, soft natural lighting, upper body, photorealistic style."

## Constraints:
- Your prompts should encourage experimentation with mixed media, introduce modern elements, and innovate shapes, colors, and compositions.
- Your prompts should prompt users to deeply reflect on and explore the possibilities of the given subject, rather than just surface-level depiction.
- You must optimize based on the content provided by the user, providing a complete subject, environment, composition, style, and image setting.
- Your prompt words should only have prompt words and no other system dialogues, and be output in English.
"""



class Ollama_prompt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        models = get_ollama_models()
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,"default": "小猫"}),
                "model": (models,),
                },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "Ollama_prompt"
    CATEGORY = "Achen节点"

    def Ollama_prompt(self, prompt, model ,):

        url= "http://127.0.0.1:11434"
        system=f"{system_content}"
        client = Client(host=url)
        options = {
            "temperature": 0.5,
        }        
        response = client.generate(model = model, prompt = prompt, system=system , options=options )
        response = response['response']

        # 返回翻译结果
        return (clean_text(response),)