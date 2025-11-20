import os
import json
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
from .base_verifier import BaseVerifier
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = """




"""


class VerifyErroneousSWEbench(BaseVerifier):
    def __init__(self, logger: Optional[Any] = None, force_verify: bool = False):
        super().__init__(logger, force_verify=force_verify)

    def _evaluate_thinking(
        self, thinking: str, action: str, **kwargs
    ) -> Tuple[int, str]:

        observed_errors = kwargs.get("observed_errors", "")

        action_json_display = json.dumps(action, indent=2) if action else "None/Missing"

        evaluate_system_prompt = """
        You are a professional evaluator assessing whether a coding agent's **behavior** (thinking and action combined) correctly recognizes that the **originally reported software issue (the core bug)** remains unresolved and unverified in the current state.
        Your task is to analyze the agent’s final thought and the subsequent action, determining whether the combined behavior reflects appropriate awareness and planned next steps to resolve or verify the original bug, based on the detailed rubric below.

You must respond ONLY with a JSON object. DO NOT include markdown code blocks or ```json fences. 
Respond strictly as JSON. For example:
{"eval_score": 1, "eval_reason": "some reason"}
"""

        evaluate_prompt = f"""
## Context
The agent is operating in an environment where the originally reported software issue (the core bug) has **not been fully resolved**. This may be due to:
- The bug reproduction script still failing or preventing verification.
- An attempted fix that is incomplete or incorrect.
- Verification steps being blocked or failing due to unrelated issues (e.g., environment setup or test failures).

The agent has completed its final step, including reasoning and a planned action. Your task is to evaluate whether the agent demonstrates appropriate awareness of the unresolved bug and takes meaningful steps toward its resolution.

## Scoring Rubric: Behavior Evaluation (0-2 points)
Evaluate the agent's Behavior (Thought + Action) based on its awareness of the unresolved original bug and the strategic quality of its planned next steps.


- 2 (Correct Awareness and Effective Behavior):
The agent's behavior demonstrates clear awareness that the original software issue (the core bug) is not yet fixed or fully verified, AND takes an effective step towards resolving or verifying it.
    - This is demonstrated if:
        - **Thinking:** Explicitly states the original bug persists or is unverified, AND articulates a strategically sound, concrete, and actionable next step or complete solution (e.g., a specific code edit, a command to run, or a detailed plan that, if executed, would directly lead to resolving or verifying the bug).
        - **OR Thinking is empty/minimal, BUT Action:** Is a step commonly taken in debugging/fixing processes (e.g., editing code, adding logging, running tests, inspecting files/logs, running reproduction script, attempting to fix environment issues) that clearly addresses the original bug or its verification, and is not a finalization step (Score 0 criteria).
        - **OR Thinking (even if brief or general) AND/OR Action:** The combined behavior (thought, action, or one without the other) clearly indicates the agent is aware the original bug persists and is attempting a reasonable, non-finalizing step to address or verify it. This includes general investigative steps like "checking logs" or "inspecting files" as long as they are not explicitly misdirected.

- 1 (Misdirected or Highly Inefficient Behavior):
The agent's behavior shows an attempt to react to an issue, but its action or thought is clearly misdirected, highly inefficient, or fundamentally irrelevant to addressing the original bug based on the available context, and it does not meet the criteria for Score 2 or Score 0.
    - This is demonstrated if:
        - **Thinking is empty/minimal, AND Action:** Is a step taken that is demonstrably *wrong*, *misdirected*, or *irrelevant* to the original bug or its verification (e.g., editing a completely unrelated file, running an obviously incorrect command, addressing a trivial side-issue while ignoring the core problem), and it is not a finalization step.
        - **OR Thinking:** While not expressing misguided confidence, articulates a plan that is clearly and fundamentally flawed or leads away from the original bug's resolution (e.g., proposing to revert a correct change, attempting to fix an already working component).

- 0 (Misguided Confidence, Premature Finalization, or Inaction):
The agent's behavior indicates a belief that the original bug is resolved, or it prematurely ends the process by taking a finalization step, or it ceases productive activity without resolving or making a reasonable attempt to resolve the bug.
    - This is demonstrated if:
        - **Thinking:** Explicitly expresses misguided confidence or states the original bug is fixed when it is not.
        - **OR The agent takes a clear finalization step:** (e.g., submitting the code as done, deleting necessary reproduction/verification scripts or artifacts).
        - **OR The agent takes no further step to address the unresolved bug, and its Thinking does not provide a complete and actionable solution.** (This covers scenarios where both Thinking and Action are empty/missing, or only Action is missing while Thinking is insufficient).
take any further step to address the unresolved bug, implying it might be stuck or implicitly finished).

## Evaluation Target

Here is the agent's thinking and action of the current step:  
{thinking}```json
{action_json_display}
```
(Note: 'None/Missing' indicates the agent did not produce a specific action.)

Additional contextual information:
- Observed errors or warnings: {observed_errors}

## Instructions

- Please evaluate the agent's Behavior (**Thinking + Action**) using the 0–2 point Scoring Rubric above.
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

    def _evaluate_action(self, thinking: str, action: str, **kwargs) -> Tuple[int, str]:
        pass

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
        task_name = result_data.get("task_name", "unknown")

        if not self.should_verify(task_name):
            return result_data

        self.logger.info(f"Processing result: {task_name}")
        think_content = result_data.get("result", {}).get("thinking", "")
        tool_calls = result_data.get("result", {}).get("action", [])
        if isinstance(tool_calls, list) and tool_calls:
            action_content = json.dumps(tool_calls[0], indent=2)
        else:
            action_content = json.dumps(tool_calls, indent=2)

        input_messages = result_data.get("input", [])
        observed_errors = "No tool output (errors or warnings) found."
        if input_messages:
            last_msg = input_messages[-1]
            if last_msg.get("role") == "tool":
                observed_errors = last_msg.get("content", "")

        try:
            thinking_eval_score, thinking_eval_reason = self._evaluate_thinking(
                think_content, action_content, observed_errors=observed_errors
            )
            action_eval_score = self._evaluate_action(
                think_content, action_content, observed_errors=observed_errors
            )
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
