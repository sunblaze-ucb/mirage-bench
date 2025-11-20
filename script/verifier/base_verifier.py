#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from enum import Enum
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed


class BaseVerifier(ABC):
    """
    Abstract base class for all verifiers.
    This class defines the common structure and required methods for verifiers.
    """

    # 评分枚举类
    class Score(int, Enum):
        ZERO = 0
        ONE = 1
        TWO = 2

    # 思维评估模型
    class EvaluateThinking(BaseModel):
        eval_score: "BaseVerifier.Score"
        eval_reason: str

    # 行动评估模型
    class EvaluateAction(BaseModel):
        eval_score: "BaseVerifier.Score"
        eval_reason: str

    DEFAULT_MODEL_NAME = "o4-mini"
    DEFAULT_MODEL_TEMPERATURE = 0.0
    DEFAULT_RESULT_FIELD_NAME = "verified_result"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        force_verify: bool = False,
        model_name: Optional[str] = None,
        model_temperature: Optional[float] = None,
        result_field_name: Optional[str] = None,
    ):
        """
        Initialize the verifier.

        Args:
            logger: Optional logger
            force_verify: Whether to force verification even if result file exists
        """
        self.inference_results = None
        self.verified_results = []
        self.output_dir = None
        self.force_verify = force_verify
        self.model_name = model_name or getattr(
            self, "DEFAULT_MODEL_NAME", BaseVerifier.DEFAULT_MODEL_NAME
        )
        default_temp = getattr(
            self,
            "DEFAULT_MODEL_TEMPERATURE",
            BaseVerifier.DEFAULT_MODEL_TEMPERATURE,
        )
        self.model_temperature = (
            model_temperature if model_temperature is not None else default_temp
        )
        self.result_field_name = result_field_name or getattr(
            self,
            "DEFAULT_RESULT_FIELD_NAME",
            BaseVerifier.DEFAULT_RESULT_FIELD_NAME,
        )

        # Use provided logger or create a default one with class name
        if logger:
            self.logger = logger.getChild(self.__class__.__name__)
        else:
            self.logger = self._setup_default_logger()

    def _setup_default_logger(self) -> logging.Logger:
        """Set up a default logger"""
        # Use full class name to distinguish between base class and subclasses
        logger_name = self.__class__.__module__ + "." + self.__class__.__name__
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Log format with class name
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # Add handler if not already added
        if not logger.handlers:
            logger.addHandler(console_handler)

        return logger

    @abstractmethod
    def _process_inference_results(self) -> Dict[str, Any]:
        """
        Process inference results and return verified results.
        This method must be implemented by subclasses.

        Returns:
            Dictionary of verified results
        """
        pass

    @abstractmethod
    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        """
        Evaluate the model's action part

        Args:
            thinking: Thinking content
            action: Action content
            **kwargs: Additional parameters for evaluation

        Returns:
            Tuple containing score and evaluation reason
        """
        pass

    @abstractmethod
    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:
        """
        Evaluate the model's thinking part

        Args:
            thinking: Thinking content
            action: Action content
            **kwargs: Additional parameters for evaluation

        Returns:
            Tuple containing score and evaluation reason
        """
        pass

    @abstractmethod
    def _process_single_result(
        self, result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single inference result for concurrent processing
        """
        pass

    def load_inference_results(self, results_dir: str, scenario: str) -> None:
        """
        Load inference results from the given directory, filtering out invalid results.

        Args:
            results_dir: Path to the directory containing result files
        """
        self.logger.info(f"Loading inference results from {results_dir}")
        results = []
        skipped_results = 0
        total_files = 0
        
        for file in os.listdir(results_dir):
            if file.endswith(".json") and scenario in file:
                total_files += 1
                try:
                    with open(os.path.join(results_dir, file), "r") as f:
                        data = json.load(f)
                    
                    # Check if this result should be filtered out
                    if self._should_skip_result(data, file):
                        skipped_results += 1
                        continue
                        
                    results.append(data)
                    
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    self.logger.warning(f"Failed to load result file {file}: {e}")
                    skipped_results += 1
                    continue

        self.inference_results = results
        self.logger.info(f"Loaded {len(results)} valid inference results")
        if skipped_results > 0:
            self.logger.info(f"Skipped {skipped_results} invalid results out of {total_files} total files")

    def _should_skip_result(self, data: dict, filename: str) -> bool:
        """
        Check if a result should be skipped due to parsing errors or other issues.
        
        Args:
            data: The loaded JSON data
            filename: Name of the file being processed
            
        Returns:
            True if the result should be skipped, False otherwise
        """
        try:
            result = data.get('result', {})
            
            # Skip results with parse errors
            if result.get('status') == 'parse_failed':
                self.logger.debug(f"Skipping {filename}: parse failed")
                return True
                
            if 'parse_error' in result:
                self.logger.debug(f"Skipping {filename}: contains parse error")
                return True
                
            # Skip results with errors (API errors, etc.)
            if 'error' in result:
                error_msg = str(result.get('error', '')).lower()
                # Skip fatal errors but allow normal processing errors
                if any(fatal_indicator in error_msg for fatal_indicator in [
                    'does not exist', 'notfounderror', 'error code: 404',
                    'unauthorized', 'error code: 401', 'invalid api key',
                    'permission denied', 'error code: 403'
                ]):
                    self.logger.debug(f"Skipping {filename}: fatal error - {result.get('error')}")
                    return True
                    
            # Skip results without proper completion content
            if not result.get('completion') and not result.get('thinking') and not result.get('action'):
                self.logger.debug(f"Skipping {filename}: missing completion content")
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"Error checking result {filename}: {e}")
            return True  # Skip results that can't be properly checked

    def set_output_dir(self, output_dir: str) -> None:
        """
        Set the output directory.
        """
        self.output_dir = output_dir
        self.logger.info(f"Output directory set to {output_dir}")

    def save_results(self, results: Dict[str, Any] = None) -> None:
        """
        Save verified results to the output file.

        """

        output_file_name = results.get("task_name", "unknown_task")
        # output_file_name = results.get("file_name", "unknown_file")
        if output_file_name is "unknown_file":
            self.logger.info(results.keys())

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        output_file = os.path.join(self.output_dir, output_file_name + ".json")

        self.logger.info(f"Saving results to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info("Results saved successfully")

    def should_verify(self, task_name: str) -> bool:
        """
        Check if a task should be verified based on whether output file exists
        and force_verify setting.

        Args:
            task_name: The name of the task to check

        Returns:
            Whether the task should be verified
        """
        if self.force_verify:
            return True

        output_file = os.path.join(self.output_dir, task_name + ".json")
        if os.path.exists(output_file):
            self.logger.info(
                f"Skipping verification for {task_name} as result file already exists"
            )
            return False

        return True

    def _process_inference_results(self) -> None:
        self.logger.info(
            f"Processing {len(self.inference_results)} inference results with parallel processing (max workers: 20)"
        )

        # Create a ThreadPoolExecutor with 20 workers
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
            future_to_result = {
                executor.submit(self._process_single_result, result_data): result_data
                for result_data in self.inference_results
            }

            # Process results as they complete
            for future in tqdm(
                as_completed(future_to_result), total=len(self.inference_results)
            ):
                result = future.result()
                if result:
                    self.verified_results.append(result)

    def __call__(self) -> Dict[str, Any]:
        """
        Execute verification and return verified results.

        Returns:
            Dictionary of verified results
        """
        # Check if inference results are loaded
        if self.inference_results is None:
            raise ValueError(
                "Inference results are not loaded. Call load_inference_results first."
            )

        # Only verify if results are None or first call
        self.logger.info("Starting verification of inference results")
        self._process_inference_results()
        self.logger.info("Verification completed")

        return self.verified_results
