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

    def __init__(
        self, logger: Optional[logging.Logger] = None, force_verify: bool = False
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
        Load inference results from the given directory.

        Args:
            results_dir: Path to the directory containing result files
        """
        self.logger.info(f"Loading inference results from {results_dir}")
        results = []
        for file in os.listdir(results_dir):
            if file.endswith(".json") and scenario in file:
                with open(os.path.join(results_dir, file), "r") as f:
                    data = json.load(f)
                    results.append(data)

        self.inference_results = results
        self.logger.info(f"Loaded {len(results)} inference results")

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
