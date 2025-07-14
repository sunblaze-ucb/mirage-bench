import os
import json


def clear_repetitive_send_msg_files(root_dir):
    """
    Recursively search for all json files in the given directory.
    Delete files where the 'repetitive_action' key contains 'send_msg_to_user'.
    Print the paths of deleted files.
    """
    deleted_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if (
                        "repetitive_action" in data
                        and "send_msg_to_user" in data["repetitive_action"]
                    ):
                        os.remove(file_path)
                        deleted_files.append(file_path)
                        print(f"Deleted: {file_path}")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error processing {file_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error with {file_path}: {e}")

    print(f"Total files deleted: {len(deleted_files)}")
    return deleted_files


if __name__ == "__main__":
    root_directory = "/home/weichenzhang/hallucination/llm_agent_hallucination_data/inferenced_results/repetitive_4"
    clear_repetitive_send_msg_files(root_directory)
