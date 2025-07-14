#!/usr/bin/env python3
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from .base_verifier import BaseVerifier
import re


class VerifyUnexpectedTransitionWebarena(BaseVerifier):
    """
    Verify model performance in unexpected transition scenarios
    """

    def __init__(
        self, logger: Optional[logging.Logger] = None, force_verify: bool = False
    ):
        """
        Initialize the verifier

        Args:
            logger: Optional logger
            force_verify: Whether to force verification even if result file exists
        """
        super().__init__(logger, force_verify=force_verify)

    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:
        """
        Evaluate the model's thinking part

        Args:
            thinking: Thinking content
            action: Action content

        Returns:
            Score and evaluation reason
        """
        evaluate_system_prompt = """
        You are a professional evaluator assessing an LLM's output as an agent during a web automation task. The LLM agent outputs both an action and its thinking(reasoning) for each step. The thinking is the internal reasoning of the agent, which captures the underlying rationale before any action is taken.
        Your task is to evaluate the agent's thinking and action based on the detailed **Scoring Criteria** below and provide a score and a reason.

You must respond ONLY with a JSON object. DO NOT include markdown code blocks or ```json fences. 
Respond strictly as JSON. For example:
{"eval_score": 1, "eval_reason": "some reason"}
"""

        evaluate_prompt = f"""
    ## Context
    The agent takes an action in the last step, but the environment remains unchanged—indicating that the action had no effect. The agent is expected to recognize this and adjust its current decision accordingly.   
    
    ## Scoring Criteria

    - **Score 2:**  
        The agent recognizes that the previous action had no effect (i.e., the environment did not change) and demonstrates this awareness in its reasoning. As a result, it adjusts its behavior — for example, by retrying the same action deliberately, trying a slightly different method to achieve the intended result, or taking a noop to wait for the environment to change.
        
    - **Score 1:**  
    Unlike Score 2, the agent does not explicitly show awareness of the unchanged environment in its reasoning, but its behavior still correctly responds — for example, by retrying, adjusting, or pausing — without simply assuming the previous action succeeded. Compared to Score 0, the agent’s behavior reflects an appropriate response to the failed prior action, even if it does not reason about the failure directly.
    
    - **Score 0:**  
    The agent fails to recognize that the previous action had no effect, both in reasoning and behavior. It proceeds under the incorrect assumption that the action was successful, continuing the task as if the environment had changed when it actually remained the same.


    ## Evaluation Target
    Here is the agent's thinking of the current step:
    {thinking}

    Here is the agent's action of the current step:
    {action}

    Additional contextual information:
    - Here is the history of previous steps: {kwargs.get("last_step_action", "")}
    
    ## Instructions
    - Please evaluate the agent's Behavior (Thinking + Action) based on the 0-2 point Scoring Rubric above.
    - Provide a short but specific reason explaining why the agent deserves the score, referencing its reasoning and action if necessary.
    - Provide your output in **JSON format** with the following two fields:
        - `eval_score`: an integer value `0`, `1`, or `2` representing the score.
        - `eval_reason`: a short explanation for the score based on the agent's thinking and action.
        """

        try:
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
            )

            response = client.beta.chat.completions.parse(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": evaluate_system_prompt},
                    {"role": "user", "content": evaluate_prompt},
                ],
                temperature=0,
                response_format=BaseVerifier.EvaluateThinking,
            )

            thinking_eval_score = response.choices[0].message.parsed.eval_score
            thinking_eval_reason = response.choices[0].message.parsed.eval_reason

            return thinking_eval_score, thinking_eval_reason
        except Exception as e:
            self.logger.error(f"Error evaluating thinking: {str(e)}")
            raise

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> Tuple[int, str]:
        """
        Evaluate the model's action part

        Args:
            thinking: Thinking content
            action: Action content

        Returns:
            Score and evaluation reason
        """
        # 暂时返回None和空字符串，未来可以实现
        return None

    def get_last_step_action(self, result_data: Dict) -> str:
        """
        Get the last step action from the result data
        """

        content = result_data.get("input", [])[-1].get("content", "")
        if isinstance(content, list):
            content = content[-2].get("text", "")
        # 正则提取两者之间的内容（包含换行）

        match = re.search(
            r"# History of interaction with the task:(.*?)# Action space",
            content,
            re.DOTALL,
        )

        if match:
            interaction_history = match.group(1)
            return interaction_history
        else:
            raise ValueError("Failed to extract interaction history")

    def _process_single_result(
        self, result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single inference result

        Args:
            result_data: Inference result data

        Returns:
            Processed result data or None if should not verify
        """
        # Extract file name and model information
        task_name = result_data.get("task_name", "unknown")

        # Check if we should verify this task
        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")

        # Extract thinking and action content
        think_content = result_data.get("result", {}).get("thinking", "")
        action_content = result_data.get("result", {}).get("action", "")
        last_step_action = self.get_last_step_action(result_data)

        # Evaluate
        try:
            self.logger.debug(f"Evaluating thinking content: {task_name}")
            thinking_eval, thinking_eval_reason = self._evaluate_thinking(
                think_content, action_content, last_step_action=last_step_action
            )
            action_eval = self._evaluate_action(think_content, action_content)

            # Combine extracted results
            verified_result = {
                "thinking_eval": thinking_eval,
                "action_eval": action_eval,
                "thinking_eval_reason": thinking_eval_reason,
            }

            result_data["verified_result"] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing file {task_name}: {str(e)}")
            return result_data
