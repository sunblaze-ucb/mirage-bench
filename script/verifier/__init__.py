"""
Verifier package for hallucination detection in LLM agents.
Contains base verifier class and specific implementation for different verification scenarios.
"""

from .base_verifier import BaseVerifier
from .verify_repetitive import VerifyRepetitive
from .verify_repetitive_swebench import VerifyRepetitiveSWEbench
from .verify_repetitive_osworld import VerifyRepetitiveOSWorld
from .verify_unexpected_transition_tac import VerifyUnexpectedTransitionTAC
from .verify_unexpected_transition_webarena import VerifyUnexpectedTransitionWebarena
from .verify_unexpected_transition_osworld import VerifyUnexpectedTransitionOSWorld
from .verify_user_questions_tac import VerifyUsersQuestionsTAC
from .verify_user_questions_taubench import VerifyUsersQuestionsTaubench
from .verify_popup import VerifyPopup
from .verify_underspecified_webarena import VerifyUnderspecifiedWebarena
from .verify_underspecified_osworld import VerifyUnderspecifiedOSWorld
from .verify_unachievable_webarena import VerifyUnachievableWebarena
from .verify_erroneous_webarena import VerifyErroneousWebarena

from .verify_erroneous_swebench import VerifyErroneousSWEbench
from .verify_misleading_swebench import VerifyMisleadingSWEbench
from .verify_misleading_webarena import VerifyMisleadingWebarena

__all__ = [
    "BaseVerifier",
    "VerifyUnexpectedTransitionTAC",
    "VerifyUnexpectedTransitionWebarena",
    "VerifyUnexpectedTransitionOSWorld",
    "VerifyRepetitive",
    "VerifyRepetitiveOSWorld",
    "VerifyUsersQuestionsTAC",
    "VerifyUsersQuestionsTaubench",
    "VerifyPopup",
    "VerifyUnderspecifiedWebarena",
    "VerifyUnderspecifiedOSWorld",
    "VerifyMisleadingWebarena",
    "VerifyUnachievableWebarena",
    "VerifyErroneous",
    "VerifyRepetitiveSWEbench",
    "VerifyErroneousSWEbench",
    "VerifyErroneousWebarena",
    "VerifyMisleadingSWEbench",
]
