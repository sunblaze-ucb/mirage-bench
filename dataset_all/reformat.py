#!/usr/bin/env python3
import os
import json
from pathlib import Path


def reformat_json_files():
    """
    Process all JSON files in the current directory:
    1. Remove 'task_seed' and 'observation_settings' fields
    2. Rename 'task_name' to 'base_task_name'
    3. Add a new 'task_name' field with the JSON filename as its value
    4. Rename files with '_01.json' suffix to '.json'
    """
    # Get the base directory
    base_dir = Path("llm_agent_hallucination_data/dataset_new")

    # Find all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"No subdirectories found in {base_dir}")
        return

    print(f"Found {len(subdirs)} subdirectories to process")
    total_files = 0
    renamed_files = 0

    # Process each subdirectory
    for subdir in subdirs:
        # Find all JSON files recursively in the current subdirectory
        json_files = list(subdir.glob("**/*.json"))
        total_files += len(json_files)

        print(f"Processing {len(json_files)} JSON files in {subdir.name}")

        for json_file in json_files:
            try:
                # Rename files with '_01.json' suffix to '.json'
                if json_file.name.endswith("_01.json"):
                    new_name = (
                        json_file.stem[:-3] + ".json"
                    )  # Remove '_01' from the stem and add '.json'
                    new_path = json_file.parent / new_name
                    json_file.rename(new_path)
                    renamed_files += 1
                    print(f"Renamed: {json_file.name} -> {new_name}")
                    # Update json_file to point to the new path
                    json_file = new_path

                # Read the JSON file
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Rename 'task_name' to 'base_task_name' if it exists
                # if 'task_name' in data:
                #     data['base_task_name'] = data['task_name']

                # Add new 'task_name' with the filename without the extension
                data["task_name"] = json_file.stem

                # Remove specified fields
                if "task_seed" in data:
                    del data["task_seed"]

                if "observation_settings" in data:
                    del data["observation_settings"]

                if "model_config" in data:
                    del data["model_config"]

                if "max_steps" in data:
                    del data["max_steps"]

                if "base_task_name" in data:
                    del data["base_task_name"]

                if "agent_name" in data:
                    del data["agent_name"]

                if "goal" in data and data["goal"] is None:
                    data["goal"] = ""

                # Add 'goal' field with null value if it doesn't exist
                if "goal" not in data:
                    data["goal"] = None

                if "repetitive_action" not in data:
                    data["repetitive_action"] = ""

                # 统一字段顺序
                ordered_data = {}
                # 定义字段顺序
                field_order = [
                    "task_name",
                    "agent_name",
                    "goal",
                    "input_step",
                    "input",
                    "repetitive_action",
                ]

                # 按照定义的顺序重新组织数据
                for field in field_order:
                    if field in data:
                        ordered_data[field] = data[field]

                # 添加任何未在预定义顺序中但存在于原始数据中的字段
                for key in data:
                    if key not in ordered_data:
                        ordered_data[key] = data[key]

                # 用有序的数据替换原始数据
                data = ordered_data
                # Write the modified data back to the file
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"Processed: {json_file.relative_to(base_dir)}")

            except Exception as e:
                print(f"Error processing {json_file.name}: {str(e)}")

    print(f"Reformatting complete! Processed {total_files} JSON files in total.")
    print(f"Renamed {renamed_files} files from '*_01.json' to '*.json'.")


if __name__ == "__main__":
    reformat_json_files()
