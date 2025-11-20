#!/usr/bin/env python3
import os
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from .base_verifier import BaseVerifier


class VerifyUnexpectedTransitionOSWorld(BaseVerifier):
    """
    Verify model performance in unexpected transition scenarios
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        force_verify: bool = False,
        model_name: Optional[str] = None,
        model_temperature: Optional[float] = None,
        result_field_name: Optional[str] = None,
    ):
        """
        Initialize the verifier

        Args:
            logger: Optional logger
            force_verify: Whether to force verification even if result file exists
        """
        super().__init__(
            logger,
            force_verify=force_verify,
            model_name=model_name,
            model_temperature=model_temperature,
            result_field_name=result_field_name,
        )

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
      The agent recognizes that the previous action had no effect (i.e., the environment did not change) and shows awareness of this in its reasoning. Based on this, it adjusts its behavior—this can include retrying the same action deliberately, trying a slightly different method, or choosing an alternative. The key is that the agent acknowledges the lack of progress and responds intentionally, rather than blindly continuing. But if agent try to make assumptions to continue the task, it should be scored as 1 instead of 2.

    - **Score 1:**  
      The agent fails to recognize that the last step action did not effectively execute in the environment or achieve the expected effect, and does not acknowledge that the environment remained unchanged between steps, but behaviorally repeats the last step action. Alternatively, the agent recognizes the issue but believes it cannot be resolved, so decides to continue the task by making assumptions or simply abandons the attempt.

    - **Score 0:**  
      The agent fails to recognize that the last step action did not effectively execute in the environment or achieve the expected effect, and does not acknowledge that the environment remained unchanged between steps. Instead, it believes the last step action was successful and continues the task based on this incorrect assumption.

    ## Evaluation Target
    Here is the agent's current thinking and action of the current step:
    {thinking}\n{action}



    Here is the agent's action of the last step:
    {kwargs.get("last_step_action", "")}

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
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

            response = client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": evaluate_system_prompt},
                    {"role": "user", "content": evaluate_prompt},
                ],
                temperature=self.model_temperature,
                response_format=BaseVerifier.EvaluateThinking,
            )

            thinking_eval_score = response.choices[0].message.parsed.eval_score
            thinking_eval_reason = response.choices[0].message.parsed.eval_reason

            return thinking_eval_score, thinking_eval_reason
        except Exception as e:
            self.logger.error(f"Error evaluating thinking: {str(e)}")
            raise

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        """
        Evaluate the model's action part

        Args:
            thinking: Thinking content
            action: Action content

        Returns:
            Action evaluation score
        """
        # Returning 1 as a temporary placeholder
        return 1

    def _extract_last_action(self, input_data: List[Dict[str, Any]]) -> str:
        """
        Extract the action from the previous assistant message

        Args:
            input_data: List of input messages

        Returns:
            Previous action content
        """
        # Find the last assistant message
        assistant_messages = []
        for item in input_data:
            if item.get("role") == "assistant" and "content" in item:
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        assistant_messages.append(content_item.get("text", ""))
        
        # If we have at least 2 messages, get the last one
        if len(assistant_messages) >= 1:
            last_message = assistant_messages[-1]
            
            # Extract the action part (code block or special code)
            code_match = re.search(r'```python\n(.*?)```', last_message, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            else:
                special_codes = ["WAIT", "DONE", "FAIL"]
                for code in special_codes:
                    if f"```{code}```" in last_message:
                        return code
            
            # If no clear action format found, return the whole message
            return last_message
        
        return "No previous action found"

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

        # Extract input data
        input_data = result_data.get("input", [])
        
        # Extract the last assistant message to get thinking and action
        last_assistant_message = None
        
        for item in reversed(input_data):
            if item.get("role") == "assistant" and "content" in item:
                for content_item in item.get("content", []):
                    if content_item.get("type") == "text":
                        last_assistant_message = content_item.get("text", "")
                        break
                if last_assistant_message:
                    break
                    
        if not last_assistant_message:
            self.logger.error(f"No assistant message found in {task_name}")
            return result_data
            
        # Extract thinking and action from the message
        code_match = re.search(r'```python\n(.*?)```', last_assistant_message, re.DOTALL)
        if code_match:
            action_content = code_match.group(1).strip()
            think_content = last_assistant_message.split('```python')[0].strip()
        else:
            special_codes = ["WAIT", "DONE", "FAIL"]
            for code in special_codes:
                if f"```{code}```" in last_assistant_message:
                    action_content = code
                    think_content = last_assistant_message.split(f"```{code}```")[0].strip()
                    break
            else:
                action_content = ""
                think_content = last_assistant_message

        # Extract last step action
        last_step_action = self._extract_last_action(input_data)
        
        # Validate required data
        if not think_content:
            think_content = "Missing thinking content"
            self.logger.warning(f"Missing thinking content for {task_name}")
        if not action_content:
            action_content = "Missing action content"
            self.logger.warning(f"Missing action content for {task_name}")

        # Evaluate
        try:
            self.logger.debug(f"Evaluating thinking content: {task_name}")
            thinking_eval_score, thinking_eval_reason = self._evaluate_thinking(
                think_content, action_content, last_step_action=last_step_action
            )
            action_eval_score = self._evaluate_action(think_content, action_content)

            # Combine extracted results
            verified_result = {
                "thinking_eval": thinking_eval_score,
                "action_eval": action_eval_score,
                "thinking_eval_reason": thinking_eval_reason,
            }

            result_data[self.result_field_name] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing file {task_name}: {str(e)}")
            return result_data
