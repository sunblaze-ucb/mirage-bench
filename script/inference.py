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

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
SWE_TOOLS_PATH = "swebench_tool_gpt.json"

# Import utility functions
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
    model_name: str,
    messages: List[Dict[str, Any]],
    logger: logging.Logger,
    tools: Any = None,
    max_retries: int = 3,
    max_parse_retries: int = 3,
) -> Dict[str, Any]:
    model_config = get_model_config(model_name)
    model_id = model_config["model_id"]
    temperature = model_config["temperature"]
    max_tokens = model_config["max_tokens"]
    top_p = model_config.get("top_p", 1)

    try:
        client = get_client(model_name)
    except Exception as e:
        logger.error(f"Failed to initialize client for model {model_name}: {e}")
        logger.error("Script will exit due to client initialization failure")
        raise

    logger.debug(f"Number of messages: {len(messages)}")

    current_messages = messages
    parse_attempts = 0
    last_response = None

    for retry in range(max_retries):
        try:
            logger.debug(
                f"Querying model {model_name}, attempt: {retry+1}/{max_retries}"
            )

            response = client.chat.completions.create(
                model=model_id,
                messages=current_messages,
                tools=tools,
                # tool_choice="auto" if tools else None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            try:
                if not tools:
                    completion = response.choices[0].message.content
                    last_response = completion
                    parsed_content = parse_html_tags_raise(
                        last_response, keys=("think", "action")
                    )

                    logger.debug(f"Model {model_name} query successful")
                    return {
                        "model": model_name,
                        "completion": last_response,
                        "thinking": parsed_content.get("think"),
                        "action": parsed_content.get("action"),
                    }
                else:
                    assistant_msg = response.choices[0].message.model_dump()
                    return {
                        "model": model_name,
                        "completion": assistant_msg,
                        "thinking": assistant_msg.get("content", ""),
                        "action": assistant_msg.get("tool_calls") or [],
                    }
            except ParseError as e:
                parse_attempts += 1
                if parse_attempts < max_parse_retries:
                    current_messages.append(
                        {"role": "assistant", "content": last_response}
                    )
                    feedback = (
                        str(e)
                        + "\nPlease ensure your answer is in one of these formats:"
                        + "\n1. Using <think>...</think> and <action>...</action> tags"
                        + "\n2. Using Thought:... and Action:... format"
                        + "\n3. Using <function=...>...</function> tags for your action"
                    )
                    current_messages.append({"role": "user", "content": feedback})
                    logger.warning(
                        f"Tag parsing failed: {e}. Attempt: {parse_attempts}/{max_parse_retries}. The model response is: {last_response}"
                    )
                    continue
                else:
                    logger.warning(
                        f"Tag parsing failed after {max_parse_retries} attempts for model {model_name}, skipping this task"
                    )
                    return {
                        "model": model_name,
                        "completion": last_response,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "parse_error": str(e),
                        "status": "parse_failed",
                    }

        except Exception as e:
            error_str = str(e)
            # Check for fatal errors that should cause script to exit
            if any(fatal_indicator in error_str.lower() for fatal_indicator in [
                "does not exist", "notfounderror", "error code: 404",
                "unauthorized", "error code: 401", "invalid api key",
                "permission denied", "error code: 403"
            ]):
                logger.error(f"Fatal error with model {model_name}: {e}")
                logger.error("This is a fatal error that requires immediate attention")
                logger.error("Script will exit to prevent saving incorrect results")
                raise
            
            if retry < max_retries - 1:
                sleep_time = 2**retry  # Exponential backoff
                logger.warning(
                    f"Error requesting model {model_name}: {e}. Will retry in {sleep_time} seconds..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to request model {model_name} after {max_retries} retries: {e}")
                logger.error("Script will exit due to repeated API failures")
                raise


def process_task(
    task_data: Dict[str, Any],
    model: str,
    setting_result_dir: Path,
    logger: logging.Logger,
    max_parse_retries: int,
    tools: Any = None,
) -> Dict[str, Any]:
    task_name = task_data.get("task_name")
    result = {
        "task_name": task_name,
        "model": model,
        "status": "skipped",
        "result_file": None,
    }

    if check_result_exists(task_name, model, setting_result_dir):
        logger.info(f"Result exists, skipping {task_name}")
        model_dir = setting_result_dir / model
        result["result_file"] = str(model_dir / f"{task_name}.json")
        return result

    messages = copy.deepcopy(task_data["input"])

    try:
        model_result = query_model(
            model_name=model,
            messages=messages,
            logger=logger,
            tools=tools,
            max_parse_retries=max_parse_retries,
        )

        # Handle parse failure case
        if model_result.get("status") == "parse_failed":
            logger.warning(f"Task {task_name} skipped due to parsing failure")
            result["status"] = "parse_failed"
            result["error"] = model_result.get("parse_error")
            # Still save the result for debugging purposes
            result_file = save_results(task_data, model_result, setting_result_dir, logger)
            result["result_file"] = str(result_file)
            return result

        result_file = save_results(task_data, model_result, setting_result_dir, logger)

        if "error" not in model_result:
            logger.info(f"Task {task_name} processed successfully")
            result["status"] = "success"
        else:
            logger.error(f"Task {task_name} failed: {model_result.get('error')}")
            result["status"] = "error"
            result["error"] = model_result.get("error")

        result["result_file"] = str(result_file)
        return result

    except Exception as e:
        logger.error(f"Error processing task {task_name}: {e}")
        result["status"] = "exception"
        result["error"] = str(e)
        return result


def process_setting(
    risk_setting: str,
    scenario: str,
    models: List[str],
    result_dir: Path,
    log_dir: Path,
    logger: logging.Logger,
    debug: bool = False,
    max_parse_retries: int = 3,
    parallel: int = 20,
) -> Dict[str, Dict[str, int]]:
    dataset_path = f"../dataset_all/{risk_setting}/{scenario}"

    dataset_dir = Path(dataset_path)
    setting_result_dir = result_dir / risk_setting

    setting_result_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info(f"Processing setting: {risk_setting}/{scenario}")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Results: {setting_result_dir}")
    logger.info(f"Parallel tasks: {parallel}")
    logger.info("=" * 50)

    dataset = load_dataset(dataset_dir, logger)
    if not dataset:
        logger.error("Empty dataset, skipping this setting")
        return

    if debug:
        dataset = dataset[:3]
        logger.info(f"Debug mode: Processing {len(dataset)} samples")

    swe_tools = None
    if scenario == "swebench":
        with open(SWE_TOOLS_PATH, "r", encoding="utf-8") as f:
            swe_tools = json.load(f)

    model_stats = {}
    
    for model in models:
        logger.info(f"\nTesting model: {model}")
        skipped_tasks = 0
        new_tasks = 0
        error_tasks = 0
        parse_failed_tasks = 0
        model_start_time = time.time()

        try:
            model_dir = setting_result_dir / model
            model_dir.mkdir(parents=True, exist_ok=True)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=parallel
            ) as executor:
                future_to_task = {
                    executor.submit(
                        process_task,
                        task_data,
                        model,
                        setting_result_dir,
                        logger,
                        max_parse_retries,
                        swe_tools,
                    ): task_data.get("task_name")
                    for task_data in dataset
                }

                for future in tqdm(
                    concurrent.futures.as_completed(future_to_task),
                    total=len(future_to_task),
                    desc=f"Testing {model}",
                ):
                    task_name = future_to_task[future]
                    try:
                        result = future.result()
                        if result["status"] == "skipped":
                            skipped_tasks += 1
                        elif result["status"] == "success":
                            new_tasks += 1
                        elif result["status"] == "parse_failed":
                            parse_failed_tasks += 1
                        else:
                            error_tasks += 1
                    except Exception as exc:
                        error_str = str(exc)
                        # Check for fatal errors that should terminate the script
                        if any(fatal_indicator in error_str.lower() for fatal_indicator in [
                            "does not exist", "notfounderror", "error code: 404",
                            "unauthorized", "error code: 401", "invalid api key",
                            "permission denied", "error code: 403",
                            "failed to initialize client"
                        ]):
                            logger.error(f"Fatal error in task {task_name}: {exc}")
                            logger.error("Terminating all tasks and exiting script due to fatal error")
                            # Cancel all pending futures
                            for pending_future in future_to_task:
                                pending_future.cancel()
                            raise exc
                        else:
                            logger.error(f"Task {task_name} exception: {exc}")
                            error_tasks += 1

        except KeyboardInterrupt:
            logger.warning("User interrupted, exiting early")
            raise
        except Exception as e:
            logger.error(f"Error testing model {model}: {e}")

        model_time = time.time() - model_start_time
        total_tasks = skipped_tasks + new_tasks + error_tasks + parse_failed_tasks
        logger.info(f"Model {model} completed: {total_tasks} tasks processed")
        logger.info(
            f"- Skipped: {skipped_tasks}, Success: {new_tasks}, Errors: {error_tasks}, Parse Failed: {parse_failed_tasks}"
        )
        logger.info(f"- Time: {model_time:.2f} seconds")
        
        # Store stats for this model
        model_stats[model] = {
            "skipped": skipped_tasks,
            "success": new_tasks,
            "errors": error_tasks,
            "parse_failed": parse_failed_tasks,
            "total": total_tasks,
            "time": model_time,
        }
    
    return model_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test multiple LLM models and save results"
    )
    parser.add_argument(
        "--risk-setting",
        type=str,
        default="unachievable",
        help="Risk setting folder name (e.g., 'unachievable', 'harmful')",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="workarena",
        help="Scenario folder name (e.g., 'workarena', 'shopping')",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all risk settings and scenarios in dataset_all directory",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=20,
        help="Number of parallel tasks to run (default: 20)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="../logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            # "gpt-4o-2024-11-20",
            # "gpt-4o-mini-2024-07-18",
            # "gemini-2.0-flash",
            # "claude-3-5-sonnet-20240620",
            # "deepseek-reasoner",
            # "deepseek-chat",
            # "Qwen2.5-7B-Instruct",
            # "llama-3.3-70b-chat",
            "gemini-2.5-flash",
        ],
        help="List of models to test, available models: "
        + ", ".join(get_available_models()),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Debug mode, process only a few samples",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level",
    )
    parser.add_argument(
        "--max-parse-retries",
        type=int,
        default=3,
        help="Maximum retry attempts when parsing fails",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_level = getattr(logging, args.log_level)
    log_dir = Path(args.log_dir)

    logger = setup_logger(log_dir, log_level)

    # result_dir = Path(args.result_dir)
    result_dir = Path("../inferenced_results")  # 写死结果目录
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("Starting inference task")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Parallel tasks: {args.parallel}")
    logger.info("=" * 50)

    if args.run_all:
        logger.info("Running in ALL settings mode")
        dataset_all_dir = Path("../dataset_all")
        settings = get_all_settings(dataset_all_dir, logger)

        if not settings:
            logger.error("No valid settings found, exiting")
            return

        logger.info(f"Found {len(settings)} settings to process")
        
        # Initialize overall stats for each model
        overall_model_stats = {model: {
            "skipped": 0, "success": 0, "errors": 0, "parse_failed": 0, 
            "total": 0, "time": 0.0, "settings_count": 0
        } for model in args.models}
        
        for setting in settings:
            risk_setting = setting["risk_setting"]
            scenario = setting["scenario"]

            try:
                setting_stats = process_setting(
                    risk_setting=risk_setting,
                    scenario=scenario,
                    models=args.models,
                    result_dir=result_dir,
                    log_dir=log_dir,
                    logger=logger,
                    debug=args.debug,
                    max_parse_retries=args.max_parse_retries,
                    parallel=args.parallel,
                )
                
                # Accumulate stats for each model
                for model, stats in setting_stats.items():
                    overall_model_stats[model]["skipped"] += stats["skipped"]
                    overall_model_stats[model]["success"] += stats["success"]
                    overall_model_stats[model]["errors"] += stats["errors"]
                    overall_model_stats[model]["parse_failed"] += stats["parse_failed"]
                    overall_model_stats[model]["total"] += stats["total"]
                    overall_model_stats[model]["time"] += stats["time"]
                    overall_model_stats[model]["settings_count"] += 1
                    
            except KeyboardInterrupt:
                logger.warning("User interrupted, ending processing")
                break
            except Exception as e:
                logger.error(f"Error processing {risk_setting}/{scenario}: {e}")
                continue
        
        # Print overall statistics for each model
        logger.info("\n" + "=" * 60)
        logger.info("OVERALL STATISTICS ACROSS ALL SETTINGS")
        logger.info("=" * 60)
        
        for model in args.models:
            stats = overall_model_stats[model]
            if stats["settings_count"] > 0:
                logger.info(f"\nModel: {model}")
                logger.info(f"- Settings processed: {stats['settings_count']}")
                logger.info(f"- Total tasks: {stats['total']}")
                logger.info(f"- Skipped: {stats['skipped']}, Success: {stats['success']}, Errors: {stats['errors']}, Parse Failed: {stats['parse_failed']}")
                logger.info(f"- Total time: {stats['time']:.2f} seconds")
                logger.info(f"- Average time per setting: {stats['time']/stats['settings_count']:.2f} seconds")
            else:
                logger.info(f"\nModel: {model} - No settings processed")
    else:
        setting_stats = process_setting(
            risk_setting=args.risk_setting,
            scenario=args.scenario,
            models=args.models,
            result_dir=result_dir,
            log_dir=log_dir,
            logger=logger,
            debug=args.debug,
            max_parse_retries=args.max_parse_retries,
            parallel=args.parallel,
        )
        
        # Print overall statistics for single setting mode
        logger.info("\n" + "=" * 60)
        logger.info(f"STATISTICS FOR {args.risk_setting}/{args.scenario}")
        logger.info("=" * 60)
        
        for model in args.models:
            if model in setting_stats:
                stats = setting_stats[model]
                logger.info(f"\nModel: {model}")
                logger.info(f"- Total tasks: {stats['total']}")
                logger.info(f"- Skipped: {stats['skipped']}, Success: {stats['success']}, Errors: {stats['errors']}, Parse Failed: {stats['parse_failed']}")
                logger.info(f"- Time: {stats['time']:.2f} seconds")


    logger.info("\nAll inference tasks complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupted, program exiting")
    except Exception as e:
        print(f"Error occurred during execution: {e}")
        exit(1)
