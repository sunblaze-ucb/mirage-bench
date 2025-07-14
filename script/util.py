import os
import re
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Regular expressions for extracting tags
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)
FUNCTION_PATTERN = re.compile(r"<function=.*?>(.*?)</function>", re.DOTALL)
THOUGHT_ACTION_PATTERN = re.compile(
    r"Thought:?\s*(.*?)(?:\n+|\s+)Action:?\s*(.*?)(?=$|\n\n)", re.DOTALL
)
PYTHON_CODE_BLOCK_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)

class ParseError(Exception):
    """Tag parsing error"""

    pass


def setup_logger(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"inference_{timestamp}.log"

    logger = logging.getLogger("inference")
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_dataset(dataset_dir: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    dataset = []
    logger.info(f"Starting to load dataset: {dataset_dir}")

    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        return []

    for json_file in dataset_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                info_data = json.load(f)

            if "input" not in info_data:
                logger.warning(f"Input field not found in {json_file.name}")
                raise ValueError(f"Input field not found in {json_file.name}")

            dataset.append(info_data)
            logger.debug(f"Task loaded: {json_file.name}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {json_file.name}: JSON format error - {e}")
        except Exception as e:
            logger.error(f"Error reading {json_file.name}: {e}")

    logger.info(f"Dataset loading complete, total {len(dataset)} tasks")
    return dataset


def extract_html_tags(text: str, keys: Tuple[str, ...]) -> Dict[str, List[str]]:
    content_dict = {}
    for key in keys:
        pattern = re.compile(f"<{key}>(.*?)</{key}>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def parse_html_tags(
    text: str,
    keys: Tuple[str, ...],
    optional_keys: Tuple[str, ...] = (),
    merge_multiple: bool = False,
) -> Tuple[Dict[str, str], bool, str]:
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if key not in content_dict:
            if key not in optional_keys:
                retry_messages.append(
                    f"Missing tag <{key}> in the answer. Please ensure your answer includes the <{key}> tag."
                )
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of <{key}> tag. You should only have one."
                    )
                else:
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message


def parse_html_tags_raise(
    text: str,
    keys: Tuple[str, ...],
    optional_keys: Tuple[str, ...] = (),
    merge_multiple: bool = False,
) -> Dict[str, str]:
    thought_action_match = THOUGHT_ACTION_PATTERN.search(text)
    if thought_action_match:
        think_content = thought_action_match.group(1).strip()
        action_content = thought_action_match.group(2).strip()
        return {"think": think_content, "action": action_content}
        
    function_matches = FUNCTION_PATTERN.findall(text)
    if function_matches:
        action_content = "".join(function_matches)
        think_content = text.split("<function=")[0].strip()
        return {"think": think_content, "action": action_content}        

    code_blocks = PYTHON_CODE_BLOCK_PATTERN.findall(text)
    action_content = ""
    if code_blocks:
        action_content = code_blocks[-1].strip() 
        think_content = PYTHON_CODE_BLOCK_PATTERN.sub("", text).strip()
        return {"think": think_content, "action": action_content}

    content_dict, valid, retry_message = parse_html_tags(text, keys, optional_keys, merge_multiple=merge_multiple)
    if not valid:
        raise ParseError(retry_message)
    return content_dict

def save_results(
    task_data: Dict[str, Any],
    model_result: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger,
) -> Path:
    result_data = task_data.copy()
    task_name = result_data["task_name"]
    result_data["result"] = model_result

    model_dir = output_dir / model_result["model"]
    model_dir.mkdir(parents=True, exist_ok=True)

    result_file = model_dir / f"{task_name}.json"

    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Results saved to {result_file}")
    except Exception as e:
        logger.error(f"Error saving results to {result_file}: {e}")

    return result_file


def check_result_exists(task_name: str, model: str, result_dir: Path) -> bool:
    result_file = result_dir / model / f"{task_name}.json"
    return result_file.exists()


def get_all_settings(
    dataset_all_dir: Path, logger: logging.Logger
) -> List[Dict[str, str]]:
    settings = []

    if not dataset_all_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_all_dir}")
        return []

    for risk_setting_dir in dataset_all_dir.iterdir():
        if not risk_setting_dir.is_dir() or risk_setting_dir.name.startswith("."):
            continue

        risk_setting = risk_setting_dir.name

        for scenario_dir in risk_setting_dir.iterdir():
            if not scenario_dir.is_dir() or scenario_dir.name.startswith("."):
                continue

            scenario = scenario_dir.name
            settings.append({"risk_setting": risk_setting, "scenario": scenario})

    return settings
