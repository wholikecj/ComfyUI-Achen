
import importlib

version_code = [1, 9]
version_str = f"V{version_code[0]}.{version_code[1]}" + (f'.{version_code[2]}' if len(version_code) > 2 else '')
print(f"### Loading: comfyui-Achen-Node ({version_str})")


node_list = [
    "API_Node",
    "Latent_nodes",
    #"Prompt_styler",
    "Prompt_Builder"
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(".node.{}".format(module_name), __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS" , "WEB_DIRECTORY"]


try:
    import cm_global
    cm_global.register_extension('comfyui-Achen-Node',
                                 {'version': version_code,
                                  'name': 'Prompt Builder',
                                  'nodes': set(NODE_CLASS_MAPPINGS.keys()),
                                  'description': '此扩展提供了各种节点以支持Lora Block Weight和Impact Pack。提供了许多易于应用的区域特性和应用，用于Variation Seed。', })
except:
    pass
