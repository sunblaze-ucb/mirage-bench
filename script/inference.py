import os
import json
import argparse
import logging
import time
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import concurrent.futures

# 移除全局 os.chdir，使用相对路径常量
BASE_DIR = Path(__file__).parent.absolute()
SWE_TOOLS_PATH = BASE_DIR / "swebench_tool_gpt.json"

# Import utility functions
# 假设 util 和 model_config 在同一目录下，如果不是请调整 sys.path
from util import (
    setup_logger,
    load_dataset,
    parse_html_tags_raise,
    save_results,
    check_result_exists,
    get_all_settings,
    ParseError,
)

# Import model configuration
from model_config import (
    get_client,
    get_model_config,
    get_available_models,
)


def query_model(
    client: Any,
    model_config: Dict[str, Any],
    messages: List[Dict[str, Any]],
    logger: logging.Logger,
    tools: Any = None,
    max_net_retries: int = 3,
) -> Dict[str, Any]:
    """
    执行模型查询。
    只处理网络重试，解析逻辑不再重试，而是根据 parse 结果设置 good_format。
    """
    model_id = model_config["model_id"]
    model_name = model_config.get("model_name", model_id) # Fallback
    temperature = model_config["temperature"]
    max_tokens = model_config["max_tokens"]
    top_p = model_config.get("top_p", 1)

    # 网络请求重试循环
    for retry in range(max_net_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            # --- 处理响应 ---
            
            # 1. 有 Tools 的情况
            if tools:
                assistant_msg = response.choices[0].message.model_dump()
                content = assistant_msg.get("content")
                return {
                    "model": model_name,
                    "completion": assistant_msg,
                    "thinking": content or "", # 安全处理 None
                    "action": assistant_msg.get("tool_calls") or [],
                    "good_format": True, # Tool call 只要能返回结构化数据通常视为格式正确
                }

            # 2. 无 Tools 的情况 (需要解析 Tags)
            else:
                completion_text = response.choices[0].message.content
                result = {
                    "model": model_name,
                    "completion": completion_text,
                    "thinking": None,
                    "action": None,
                    "good_format": False
                }

                try:
                    parsed_content = parse_html_tags_raise(
                        completion_text, keys=("think", "action")
                    )
                    # 解析成功
                    result["thinking"] = parsed_content.get("think")
                    result["action"] = parsed_content.get("action")
                    result["good_format"] = True
                except ParseError as e:
                    # 解析失败：不再重试，直接使用完整回复作为 action
                    logger.warning(f"Parse failed for {model_name}: {e}. Fallback to full content.")
                    result["thinking"] = ""
                    result["action"] = completion_text # Fallback
                    result["good_format"] = False 
                    result["parse_error"] = str(e) # 可选：记录具体错误

                return result

        except Exception as e:
            error_str = str(e).lower()
            # 致命错误检查
            if any(fatal in error_str for fatal in [
                "does not exist", "notfounderror", "404",
                "unauthorized", "401", "invalid api key",
                "permission denied", "403"
            ]):
                logger.error(f"Fatal error with model {model_name}: {e}")
                raise e # 直接抛出，由上层捕获并停止任务
            
            if retry < max_net_retries - 1:
                sleep_time = 2 ** retry
                logger.warning(f"Network error for {model_name}: {e}. Retry {retry+1}/{max_net_retries} in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to request model {model_name} after {max_net_retries} retries: {e}")
                return {
                    "model": model_name,
                    "status": "error",
                    "error": str(e)
                }
    
    # 理论上不应到达这里
    return {"model": model_name, "status": "error", "error": "Max retries exceeded unknown"}


def process_task(
    task_data: Dict[str, Any],
    client: Any,           # 传入已初始化的 Client
    model_config: Dict,    # 传入 Config
    setting_result_dir: Path,
    logger: logging.Logger,
    tools: Any = None,
) -> Dict[str, Any]:
    
    task_name = task_data.get("task_name")
    model_name = model_config.get("model_name_alias", model_config["model_id"]) # 假设 config 里存了别名
    
    result_info = {
        "task_name": task_name,
        "model": model_name,
        "status": "skipped",
        "result_file": None,
    }

    # 检查是否存在
    if check_result_exists(task_name, model_name, setting_result_dir):
        logger.debug(f"Result exists, skipping {task_name}")
        model_dir = setting_result_dir / model_name
        result_info["result_file"] = str(model_dir / f"{task_name}.json")
        return result_info

    messages = copy.deepcopy(task_data["input"])

    try:
        model_result = query_model(
            client=client,
            model_config=model_config,
            messages=messages,
            logger=logger,
            tools=tools,
        )

        # 保存结果
        # 注意：save_results 内部可能需要适配 good_format 字段，或者它只是 dump json
        result_file = save_results(task_data, model_result, setting_result_dir, logger)
        result_info["result_file"] = str(result_file)

        if model_result.get("status") == "error":
            result_info["status"] = "error"
            result_info["error"] = model_result.get("error")
        else:
            # 区分格式好坏，但在统计层面都算 "success" (即完成推理)
            # 如果你想统计 parse_failed，可以在这里根据 good_format 改状态
            result_info["status"] = "success"
            result_info["good_format"] = model_result.get("good_format", True)

        return result_info

    except Exception as e:
        logger.error(f"Exception processing task {task_name}: {e}")
        result_info["status"] = "exception"
        result_info["error"] = str(e)
        return result_info


def process_setting(
    risk_setting: str,
    scenario: str,
    models: List[str],
    result_dir: Path,
    log_dir: Path,
    logger: logging.Logger,
    debug: bool = False,
    parallel: int = 20,
) -> Dict[str, Dict[str, int]]:
    
    # 构建路径
    dataset_dir = BASE_DIR.parent / "dataset_all" / risk_setting / scenario
    setting_result_dir = result_dir / risk_setting
    setting_result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing setting: {risk_setting}/{scenario}")

    dataset = load_dataset(dataset_dir, logger)
    if not dataset:
        logger.error("Empty dataset, skipping this setting")
        return {}  # Fix: 返回空字典而不是 None

    if debug:
        dataset = dataset[:3]
        logger.info(f"Debug mode: Processing {len(dataset)} samples")

    swe_tools = None
    if scenario == "swebench":
        if SWE_TOOLS_PATH.exists():
            with open(SWE_TOOLS_PATH, "r", encoding="utf-8") as f:
                swe_tools = json.load(f)
        else:
            logger.warning(f"Tools file not found at {SWE_TOOLS_PATH}")

    model_stats = {}
    
    for model_name in models:
        logger.info(f"\nTesting model: {model_name}")
        
        # --- Client 初始化优化 ---
        # 在循环外初始化一次，传给所有线程
        try:
            client = get_client(model_name)
            model_config = get_model_config(model_name)
            # 确保 config 里有用来做文件夹名的 model_name
            model_config["model_name_alias"] = model_name 
        except Exception as e:
            logger.error(f"Failed to initialize client for {model_name}: {e}")
            continue

        stats = {
            "skipped": 0, "success": 0, "errors": 0, 
            "bad_format": 0, # 新增统计
            "total": 0, "time": 0.0
        }
        
        model_start_time = time.time()
        
        # 确保模型子目录存在
        (setting_result_dir / model_name).mkdir(parents=True, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            future_to_task = {
                executor.submit(
                    process_task,
                    task_data,
                    client,       # Pass client
                    model_config, # Pass config
                    setting_result_dir,
                    logger,
                    swe_tools,
                ): task_data.get("task_name")
                for task_data in dataset
            }

            try:
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_task),
                    total=len(future_to_task),
                    desc=f"Testing {model_name}",
                ):
                    task_name = future_to_task[future]
                    try:
                        res = future.result()
                        status = res["status"]
                        
                        if status == "skipped":
                            stats["skipped"] += 1
                        elif status == "success":
                            stats["success"] += 1
                            if res.get("good_format") is False:
                                stats["bad_format"] += 1
                        else:
                            stats["errors"] += 1
                            
                    except Exception as exc:
                        logger.error(f"Task {task_name} generated an exception: {exc}")
                        # 检查是否需要中断 (Fatal Errors)
                        error_str = str(exc).lower()
                        if "invalid api key" in error_str or "does not exist" in error_str:
                            logger.critical("Fatal API error detected. Stopping all tasks for this model.")
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise exc
                        stats["errors"] += 1

            except Exception as e:
                logger.error(f"Critical failure during model execution: {e}")
                # 继续下一个模型，而不是完全退出程序 (除非是 Key 错误，上面已经 raise 了)

        model_time = time.time() - model_start_time
        total_tasks = stats["skipped"] + stats["success"] + stats["errors"]
        stats["total"] = total_tasks
        stats["time"] = model_time
        
        logger.info(f"Model {model_name} finished in {model_time:.2f}s")
        logger.info(f"- Success: {stats['success']} (Bad Format: {stats['bad_format']})")
        logger.info(f"- Skipped: {stats['skipped']}, Errors: {stats['errors']}")
        
        model_stats[model_name] = stats
    
    return model_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Test multiple LLM models")
    parser.add_argument("--risk-setting", type=str, default="unachievable")
    parser.add_argument("--scenario", type=str, default="workarena")
    parser.add_argument("--run-all", action="store_true", default=False)
    parser.add_argument("--parallel", type=int, default=20)
    parser.add_argument("--log-dir", type=str, default="../logs")
    # 结果目录如果不传，默认放在上一级的 inferenced_results
    parser.add_argument("--result-dir", type=str, default=str(BASE_DIR.parent / "inferenced_results"))
    parser.add_argument("--models", type=str, nargs="+", default=["gemini-2.5-flash"])
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_dir = Path(args.log_dir)
    result_dir = Path(args.result_dir)
    
    # 确保日志目录存在
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir, log_level)

    logger.info("=" * 50)
    logger.info(f"Starting inference task. Result Dir: {result_dir}")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info("=" * 50)

    overall_stats = {m: {"total": 0, "success": 0, "bad_format": 0, "errors": 0, "skipped": 0, "time": 0.0} for m in args.models}

    if args.run_all:
        dataset_all_dir = BASE_DIR.parent / "dataset_all"
        settings = get_all_settings(dataset_all_dir, logger)

        if not settings:
            logger.error("No valid settings found.")
            return
        
        for setting in settings:
            try:
                s_stats = process_setting(
                    setting["risk_setting"],
                    setting["scenario"],
                    args.models,
                    result_dir,
                    log_dir,
                    logger,
                    args.debug,
                    args.parallel
                )
                
                # 聚合统计
                for m, stat in s_stats.items():
                    for k in overall_stats[m]:
                        if k in stat:
                            overall_stats[m][k] += stat[k]

            except KeyboardInterrupt:
                logger.warning("Interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error in setting loop: {e}")
    else:
        s_stats = process_setting(
            args.risk_setting,
            args.scenario,
            args.models,
            result_dir,
            log_dir,
            logger,
            args.debug,
            args.parallel
        )
        if s_stats:
             for m, stat in s_stats.items():
                 # 简单覆盖，因为只有一个 setting
                 overall_stats[m] = stat

    # 打印最终报告
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for m, s in overall_stats.items():
        if s["total"] > 0:
            logger.info(f"Model: {m}")
            logger.info(f"  Total: {s['total']} | Time: {s['time']:.1f}s")
            logger.info(f"  Success: {s['success']} (Bad Format: {s['bad_format']})")
            logger.info(f"  Skipped: {s['skipped']} | Errors: {s['errors']}")
        else:
            logger.info(f"Model: {m} - No tasks processed")

if __name__ == "__main__":
    main()