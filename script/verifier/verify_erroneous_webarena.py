import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from .base_verifier import BaseVerifier
import re
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed


class VerifyErroneousWebarena(BaseVerifier):
    """
    Verify model performance in erroneous generation in webarena
    """

    def __init__(
        self, logger: Optional[logging.Logger] = None, force_verify: bool = False
    ):
        super().__init__(logger, force_verify=force_verify)

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> int:
        pass

    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:
        evaluate_system_prompt = """
        You are a professional evaluator assessing an LLM agent's reasoning process during a web automation task. 
        The agent outputs both an action and its reasoning for each step.
        Please evaluate the agent's reasoning and action of the current step based on the evaluation criteria below and provide a score and a reason.

You must respond ONLY with a JSON object. DO NOT include markdown code blocks or ```json fences. 
Respond strictly as JSON. For example:
{"eval_score": 1, "eval_reason": "some reason"}
"""

        evaluate_prompt = f"""
        ## Context
        The agent took an erroneous action in the last step, and the environment returned a corresponding feedback indicating failure. In the current step, the agent provides a new reasoning and action. The goal is to evaluate whether the agent recognizes the error from the previous step and how it responds to that feedback in its current decision.

        ## Scoring Criteria

        - **Score 2:**  
        The agent **explicitly notices the feedback** from the environment in its reasoning and takes a **different or adjusted action** in response. This indicates a clear awareness of the previous error and an intentional effort to recover or correct it.

        - **Score 1:**  
        One of the following is true:  
        1. The agent **does not acknowledge** the environmental feedback in its reasoning but still takes an **alternative action** aligned with the same intention as the previous step.  
        2. The agent **acknowledges the error** but still **repeats the same action**.

        - **Score 0:**  
        One of the following is true:  
        1. The agent **does not notice the error** and proceeds with future actions under the **false assumption that the last step succeeded**.  
        2. The agent **fails to recognize the error** and takes an unrelated or illogical next action.

        ## Evaluation Target

        Here is the agent's thinking of the current step:  
        {thinking}

        Here is the agent's action of the current step:  
        {action}

        Additional contextual information:
        - Interaction history: {kwargs.get("action_history", "")}

        ## Instructions
        - Evaluate the agent's reasoning and action using the rubric above.
        - Assign a score from 0 to 2.
        - Provide a short justification that references whether the agent recognized the prior error and how it responded.
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

    def extract_repetitive_action_history(
        self, trajectory: List[Dict[str, Any]]
    ) -> str:
        """
        Extract repetitive action from trajectory
        """

        messages = trajectory[-1].get("content", "")
        message = messages[-2].get("text", "")
        # print(message)
        if not message:
            raise ValueError("No message found in trajectory")

        match = re.search(
            r"# History of interaction with the task:(.*?)# Action space",
            message,
            re.DOTALL,
        )
        if match:
            result = match.group(1).strip()
            return result
        else:
            raise ValueError("Pattern not found in message")

    def _process_single_result(
        self, result_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:

        task_name = result_data.get("task_name", "unknown")

        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")
        think_content = result_data.get("result", {}).get("thinking", "")
        action_content = result_data.get("result", {}).get("action", "")

        # Evaluate
        try:
            action_history = self.extract_repetitive_action_history(
                result_data.get("input", [])
            )
            self.logger.debug(f"Repetitive action: {action_history}")

            thinking_eval_score, thinking_eval_reason = self._evaluate_thinking(
                think_content, action_content, action_history=action_history
            )
            action_eval_score = self._evaluate_action(
                think_content, action_content, action_history=action_history
            )

            # Combine extracted results
            verified_result = {
                "thinking_eval": thinking_eval_score,
                "action_eval": action_eval_score,
                "thinking_eval_reason": thinking_eval_reason,
            }

            result_data["verified_result"] = verified_result
            self.save_results(result_data)
            return result_data

        except Exception as e:
            self.logger.error(f"Error processing file {task_name}: {str(e)}")
            return result_data
