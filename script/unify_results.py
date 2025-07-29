#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import collections
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def load_data(file_path):
    """
    Load data from a JSON file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict: Loaded data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_weighted_average(data_items):
    """
    Compute weighted average of scores.

    Args:
        data_items (list): List of (score, sample_count) tuples

    Returns:
        tuple: (weighted_average, total_samples)
    """
    if not data_items:
        return 0, 0

    total_weighted_score = sum(score * count for score, count in data_items)
    total_samples = sum(count for _, count in data_items)

    if total_samples == 0:
        return 0, 0

    return total_weighted_score / total_samples, total_samples


def unify_categories(data, metric_name):
    """
    Unify categories according to the specified mapping.

    Args:
        data (dict): Original score data
        metric_name (str): The metric to unify (e.g., "utility_score" or "hallucination_rate")

    Returns:
        dict: Unified category data
    """
    unified_data = collections.defaultdict(lambda: collections.defaultdict(dict))
    models = set()

    # 1. Unify unachievable categories
    for model in set(data["unachievable"].keys()).union(
        data["unachievable_easier"].keys()
    ):
        models.add(model)
        unachievable_items = []

        if model in data["unachievable"]:
            unachievable_items.append(
                (
                    data["unachievable"][model][metric_name],
                    data["unachievable"][model]["sample_count"],
                )
            )

        if model in data["unachievable_easier"]:
            unachievable_items.append(
                (
                    data["unachievable_easier"][model][metric_name],
                    data["unachievable_easier"][model]["sample_count"],
                )
            )

        weighted_score, total_samples = compute_weighted_average(unachievable_items)
        unified_data["Unachievable Goal"][model] = {
            metric_name: weighted_score,
            "sample_count": total_samples,
        }

    # 2. Unify underspecified and misleading as "Ill-specified Instructions"
    for model in set(data["underspecified"].keys()).union(data["misleading"].keys()):
        models.add(model)
        ill_specified_items = []

        if model in data["underspecified"]:
            ill_specified_items.append(
                (
                    data["underspecified"][model][metric_name],
                    data["underspecified"][model]["sample_count"],
                )
            )

        if model in data["misleading"]:
            ill_specified_items.append(
                (
                    data["misleading"][model][metric_name],
                    data["misleading"][model]["sample_count"],
                )
            )

        weighted_score, total_samples = compute_weighted_average(ill_specified_items)
        unified_data["Ill-specified Instructions"][model] = {
            metric_name: weighted_score,
            "sample_count": total_samples,
        }

    # 3. Unify repetitive_4 and repetitive_7 as "Flawed Interaction History (Repetitive)"
    for model in set(data["repetitive_4"].keys()).union(data["repetitive_7"].keys()):
        models.add(model)
        repetitive_items = []

        if model in data["repetitive_4"]:
            repetitive_items.append(
                (
                    data["repetitive_4"][model][metric_name],
                    data["repetitive_4"][model]["sample_count"],
                )
            )

        if model in data["repetitive_7"]:
            repetitive_items.append(
                (
                    data["repetitive_7"][model][metric_name],
                    data["repetitive_7"][model]["sample_count"],
                )
            )

        weighted_score, total_samples = compute_weighted_average(repetitive_items)
        unified_data["Flawed Interaction History (Repetitive)"][model] = {
            metric_name: weighted_score,
            "sample_count": total_samples,
        }

    # 4. Rename error_feedback to "Flawed Interaction History (Erroneous)"
    for model in data["error_feedback"]:
        models.add(model)
        unified_data["Flawed Interaction History (Erroneous)"][model] = data[
            "error_feedback"
        ][model]

    # 5. Rename users_questions to "User Queries Outside Task Boundary"
    for model in data["users_questions"]:
        models.add(model)
        unified_data["Out of Boundary Queries"][model] = data[
            "users_questions"
        ][model]

    # 6. Rename popup to "Pop-up Distractions"
    for model in data["popup"]:
        models.add(model)
        unified_data["Pop-up Distractions"][model] = data["popup"][model]

    # 7. Keep unexpected_transition as is
    for model in data["unexpected_transition"]:
        models.add(model)
        unified_data["Unexpected Environmental Transition"][model] = data["unexpected_transition"][
            model
        ]

    # 8. Calculate overall score for each model (weighted by sample counts)
    for model in models:
        overall_items = []

        for category in unified_data:
            if model in unified_data[category]:
                model_data = unified_data[category][model]
                overall_items.append(
                    (model_data[metric_name], model_data["sample_count"])
                )

        weighted_score, total_samples = compute_weighted_average(overall_items)
        unified_data["Overall"][model] = {
            metric_name: weighted_score,
            "sample_count": total_samples,
        }

    return dict(unified_data)


def save_unified_data(unified_data, output_file):
    """
    Save unified data to a JSON file.

    Args:
        unified_data (dict): Unified data
        output_file (str): Output file path
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(unified_data, f, indent=2)


def print_summary(unified_data, metric_name):
    """
    Print summary of unified data.

    Args:
        unified_data (dict): Unified data
        metric_name (str): The metric to print
    """
    print("Unified Categories:")
    for category in unified_data:
        print(f"\nCategory: {category}")

        # Sort models by metric score
        sorted_models = sorted(
            unified_data[category].items(),
            key=lambda x: x[1][metric_name],
            reverse=True,
        )

        for model, data in sorted_models:
            print(
                f"  {model}: {data[metric_name]:.4f} (Sample count: {data['sample_count']})"
            )


def main():
    # Input and output file paths
    utility_score_input_file = "../processed_results/utility_score.json"
    hallucination_rate_input_file = "../processed_results/hallucination_rate.json"
    unified_utility_score_output_file = (
        "../processed_results/unified_utility_score.json"
    )
    unified_hallucination_rate_output_file = (
        "../processed_results/unified_hallucination_rate.json"
    )

    # Load data
    utility_score_data = load_data(utility_score_input_file)
    hallucination_rate_data = load_data(hallucination_rate_input_file)

    # Unify categories
    unified_utility_score_data = unify_categories(utility_score_data, "utility_score")
    unified_hallucination_rate_data = unify_categories(
        hallucination_rate_data, "hallucination_rate"
    )

    # Save unified data
    save_unified_data(unified_utility_score_data, unified_utility_score_output_file)
    save_unified_data(
        unified_hallucination_rate_data, unified_hallucination_rate_output_file
    )
    # Print summary
    print("-" * 100)
    print("Unified Utility Score:")
    print_summary(unified_utility_score_data, "utility_score")
    print("-" * 100)
    print("Unified Hallucination Rate:")
    print_summary(unified_hallucination_rate_data, "hallucination_rate")
    print("-" * 100)
    print(f"\nUnified utility score data saved to {unified_utility_score_output_file}")
    print(f"\nUnified hallucination rate data saved to {unified_hallucination_rate_output_file}")


if __name__ == "__main__":
    main()
