import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let get_wildcards_list;
try {
	const ImpactPack = await import("../ComfyUI-Impact-Pack/impact-pack.js");
	get_wildcards_list = ImpactPack.get_wildcards_list;
}
catch (error) {}

// fallback
if(!get_wildcards_list) {
	get_wildcards_list = () => { return ["Impact Pack isn't installed or is outdated."]; }
}

// 定义一个对象 pb_cache，用于缓存不同分类的提示构建器项目数据
let pb_cache = {};

/**
 * 异步函数，用于获取指定分类的提示构建器项目数据。
 * 首先检查缓存中是否已经存在该分类的数据，如果存在则直接返回缓存数据；
 * 如果不存在，则向服务器发送请求获取数据，将数据存入缓存并返回。
 * @param {string} category - 要获取提示构建器项目的分类名称
 * @returns {Promise<Array>} - 包含提示构建器项目的数组
 */
async function get_prompt_builder_items(category) {
    // 检查缓存中是否已经存在该分类的数据
    if (pb_cache[category]) {
        // 如果存在，直接返回缓存数据
        return pb_cache[category];
    } else {
        // 如果不存在，向服务器发送请求获取数据
        let res = await api.fetchApi(`/node/prompt_builder?category=${category}`);
        // 解析响应数据为 JSON 格式
        let data = await res.json();
        // 将获取到的数据存入缓存
        pb_cache[category] = data.presets;
        // 返回获取到的数据
        return data.presets;
    }
}


app.registerExtension({
	name: "Comfy.Inspire.Prompts",

	nodeCreated(node, app) {
		// 检查当前节点的 comfyClass 是否为 "PromptBuilderss"，如果是则执行以下操作
		if(node.comfyClass == "PromptBuilderss") {
			// 从节点的 widgets 数组中查找名为 'preset' 的 widget，并将其赋值给 preset_widget 常量
			const preset_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'preset')];
			// 从节点的 widgets 数组中查找名为 'category' 的 widget，并将其赋值给 category_widget 常量
			const category_widget = node.widgets[node.widgets.findIndex(obj => obj.name === 'category')];

			// 为 preset_widget 的 options 对象的 values 属性定义 getter 和 setter 方法
			Object.defineProperty(preset_widget.options, "values", {
				// setter 方法为空，即不做任何处理
				set: (x) => {},
				// getter 方法，当获取 values 属性时执行
				get: () => {
					// 调用 get_prompt_builder_items 函数，根据 category_widget 的值获取提示构建器项目数据
					get_prompt_builder_items(category_widget.value);
					// 检查缓存中是否存在该分类的数据，如果不存在则返回 ["#PRESET"]
					if(pb_cache[category_widget.value] == undefined) {
						return ["#PRESET"];
					}
					// 如果缓存中存在该分类的数据，则返回缓存中的数据
					return pb_cache[category_widget.value];
				}
			});

			// 为 preset_widget 定义回调函数，当 preset_widget 的值发生变化时执行
			preset_widget.callback = (value, canvas, node, pos, e) => {
				// 检查节点的第5个 widget 的值是否存在，如果存在则在其后面添加 ', '
				// if(node.widgets[4].value) {
				// 	node.widgets[4].value += ', ';
				// }

				// 将 node._preset_value 按 ':' 分割成数组 y
				const y = node._preset_value.split(':');
				// 如果节点的第五个部件 widget 的值是 true，则将':'前面中文添加到第六个 widget 的值后面
				if(node.widgets[4].value == true)
					node.widgets[5].value += y[0].trim()+', ';
				// 否则将':'后面英文添加到第六个 widget 的值后面
				else
					node.widgets[5].value += y[1].trim()+', ';
			}

			// 为 preset_widget 的 value 属性定义 getter 和 setter 方法
			Object.defineProperty(preset_widget, "value", {
				// setter 方法，当设置 value 属性时执行
				set: (value) => {
					// 如果设置的值不是 "#PRESET"，则将其赋值给 node._preset_value
					if (value !== "#PRESET")
						node._preset_value = value;
				},
				// getter 方法，当获取 value 属性时返回 "#PRESET"
				get: () => { return '#PRESET'; }
			});

			// 为 preset_widget 定义 serializeValue 方法，用于序列化 widget 的值，返回 "#PRESET"
			preset_widget.serializeValue = (workflowNode, widgetIndex) => { return "#PRESET"; };
		}
	}
});




// 首先保存原始的 api.queuePrompt 函数，以便后续调用
const original_queuePrompt = api.queuePrompt;
/**
 * 自定义的 queuePrompt 函数，在原始函数调用前为 workflow 对象添加 widget_idx_map 属性。
 * 该属性用于记录特定 widget 的索引信息。
 * @param {number} number - 原始 queuePrompt 函数所需的编号参数。
 * @param {Object} options - 包含 output 和 workflow 的选项对象。
 * @param {*} options.output - 输出相关信息。
 * @param {Object} options.workflow - 工作流对象，将在其中添加 widget_idx_map 属性。
 * @returns {Promise} - 返回原始 queuePrompt 函数调用的结果。
 */
async function queuePrompt_with_widget_idxs(number, { output, workflow }) {
    // 初始化 workflow 对象的 widget_idx_map 属性，用于存储 widget 索引信息
    workflow.widget_idx_map = {};

    // 遍历 app.graph 中所有节点
    for(let i in app.graph._nodes_by_id) {
        // 获取当前节点的所有 widget
        let widgets = app.graph._nodes_by_id[i].widgets;
        // 检查当前节点是否存在 widgets
        if(widgets) {
            // 遍历当前节点的所有 widget
            for(let j in widgets) {
                // 检查当前 widget 的名称是否在指定列表中，并且类型不是 'converted-widget'
                if(['seed', 'noise_seed', 'sampler_name', 'scheduler'].includes(widgets[j].name)
                    && widgets[j].type != 'converted-widget') {
                    // 如果 workflow.widget_idx_map 中还没有当前节点的记录，则初始化一个空对象
                    if(workflow.widget_idx_map[i] == undefined) {
                        workflow.widget_idx_map[i] = {};
                    }

                    // 将当前 widget 的索引存储到 workflow.widget_idx_map 中
                    workflow.widget_idx_map[i][widgets[j].name] = parseInt(j);
                }
            }
        }
    }

    // 调用原始的 queuePrompt 函数，并返回其结果
    return await original_queuePrompt.call(api, number, { output, workflow });
}

// 将自定义的 queuePrompt 函数赋值给 api.queuePrompt，替换原始函数
api.queuePrompt = queuePrompt_with_widget_idxs;