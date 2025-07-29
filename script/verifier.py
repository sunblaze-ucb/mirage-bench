import os
import argparse
import logging
from pathlib import Path
import datetime
import json


# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def setup_logger(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logger"""
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"verify_{timestamp}.log"

    # Configure logger
    logger = logging.getLogger("verifier")
    logger.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    logger = setup_logger(log_dir="../verify_logs")

    parser = argparse.ArgumentParser(
        description="Verify hallucination in LLM agent in different domains and tasks, route to different evaluators"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="unexpected_transition",
        help="Type to verify",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="webarena",
        help="Scenario to verify",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model name to verify",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force verification even if result file exists",
    )

    args = parser.parse_args()

    inference_results_dir = Path("../inferenced_results")
    verify_results_dir = Path("../verified_results")

    inference_results_path = os.path.join(inference_results_dir, args.type, args.model)

    if not os.path.exists(inference_results_path):
        logger.error(f"Verification path does not exist: {inference_results_path}")
        return

    # Build output file path
    output_dir = os.path.join(verify_results_dir, args.type, args.model)
    os.makedirs(output_dir, exist_ok=True)

    if args.type == "unexpected_transition":
        if args.scenario == "theagentcompany":
            from verifier import VerifyUnexpectedTransitionTAC

            verifier = VerifyUnexpectedTransitionTAC(logger, force_verify=args.force)

        elif args.scenario == "webarena":
            from verifier import VerifyUnexpectedTransitionWebarena

            verifier = VerifyUnexpectedTransitionWebarena(
                logger, force_verify=args.force
            )
        elif args.scenario == "osworld":
            from verifier import VerifyUnexpectedTransitionOSWorld

            verifier = VerifyUnexpectedTransitionOSWorld(
                logger, force_verify=args.force
            )
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    elif args.type == "users_questions":
        if args.scenario == "theagentcompany":
            from verifier import (
                VerifyUsersQuestionsTAC,
            )

            verifier = VerifyUsersQuestionsTAC(logger, force_verify=args.force)
        elif args.scenario == "taubench":
            from verifier import (
                VerifyUsersQuestionsTaubench,
            )

            verifier = VerifyUsersQuestionsTaubench(logger, force_verify=args.force)
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    elif args.type == "misleading":
        if args.scenario == "swebench":
            from verifier import (
                VerifyMisleadingSWEbench,
            )

            verifier = VerifyMisleadingSWEbench(logger, force_verify=args.force)
        elif args.scenario == "webarena":
            from verifier import VerifyMisleadingWebarena

            verifier = VerifyMisleadingWebarena(logger, force_verify=args.force)
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    elif args.type == "repetitive_4" or args.type == "repetitive_7":
        if args.scenario == "webarena" or args.scenario == "workarena":
            from verifier import VerifyRepetitive

            verifier = VerifyRepetitive(logger, force_verify=args.force)
        elif args.scenario == "osworld":
            from verifier import VerifyRepetitiveOSWorld

            verifier = VerifyRepetitiveOSWorld(logger, force_verify=args.force)
        elif args.scenario == "swebench":
            from verifier import VerifyRepetitiveSWEbench

            verifier = VerifyRepetitiveSWEbench(logger, force_verify=args.force)
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    elif args.type == "popup":
        if args.scenario == "webarena" or args.scenario == "osworld":
            from verifier import VerifyPopup

            verifier = VerifyPopup(logger, force_verify=args.force)
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    elif args.type == "underspecified":
        if args.scenario == "webarena":
            from verifier import VerifyUnderspecifiedWebarena

            verifier = VerifyUnderspecifiedWebarena(logger, force_verify=args.force)
        elif args.scenario == "osworld":
            from verifier import VerifyUnderspecifiedOSWorld

            verifier = VerifyUnderspecifiedOSWorld(logger, force_verify=args.force)
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    elif args.type == "unachievable" or args.type == "unachievable_easier":
        if args.scenario == "webarena" or args.scenario == "workarena":
            from verifier import VerifyUnachievableWebarena

            verifier = VerifyUnachievableWebarena(logger, force_verify=args.force)
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    elif args.type == "error_feedback":
        if args.scenario == "webarena" or args.scenario == "workarena":
            from verifier import VerifyErroneousWebarena

            verifier = VerifyErroneousWebarena(logger, force_verify=args.force)
        elif args.scenario == "swebench":
            from verifier import VerifyErroneousSWEbench

            verifier = VerifyErroneousSWEbench(logger, force_verify=args.force)
        else:
            logger.error(f"Unsupported scenario '{args.scenario}' for type '{args.type}'")
            return

    else:
        logger.error(f"Unsupported verification type: {args.type}")
        return

    # Load inference results and run verification
    verifier.load_inference_results(inference_results_path, args.scenario)
    verifier.set_output_dir(output_dir)
    verified_results = verifier()

    # Calculate statistics
    total_files = len(verified_results)
    verified_count = sum(
        1 for result in verified_results if "verified_result" in result
    )
    skipped_count = total_files - verified_count

    # Print statistics
    logger.info(f"Total results: {total_files}")
    logger.info(f"Verified: {verified_count}")
    logger.info(f"Skipped (already existed): {skipped_count}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
