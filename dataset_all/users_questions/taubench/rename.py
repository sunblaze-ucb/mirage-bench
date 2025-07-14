import os
import json

# 设置目标文件夹路径
folder_path = "/home/weichenzhang/hallucination/llm_agent_hallucination_data/dataset_all/users_questions/taubench"  # 替换为你的目录路径

# 对所有 json 文件的遍历，修改input数组中消息的tool_calls和function_call
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # 检查文件是否包含input数组
            if "input" in data and isinstance(data["input"], list):
                for message in data["input"]:
                    if isinstance(message, dict):
                        # 删除tool_calls和function_call键，无论值是什么
                        if "tool_calls" in message:
                            del message["tool_calls"]
                        if "function_call" in message:
                            del message["function_call"]

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"处理文件: {filename}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
