import os
import json
import glob
from collections import defaultdict
from pathlib import Path


def calculate_utility_scores(
    base_directory="llm_agent_hallucination_data/verified_results",
):
    """
    Calculate utility scores for all models across all settings.

    Args:
        base_directory: Base path to the verified results directory

    Returns:
        Dictionary mapping settings to model utility scores
    """
    # Dictionary to store results for each setting and model
    all_results = {}

    # Find all setting directories
    setting_dirs = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]

    for setting in setting_dirs:
        setting_path = os.path.join(base_directory, setting)
        all_results[setting] = {}

        # Find all model subdirectories in this setting
        model_dirs = [
            d
            for d in os.listdir(setting_path)
            if os.path.isdir(os.path.join(setting_path, d))
        ]

        for model in model_dirs:
            model_dir = os.path.join(setting_path, model)
            utility_values = []

            # Find all JSON files in the model directory
            json_files = glob.glob(os.path.join(model_dir, "*.json"))

            for json_file in json_files:
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)

                    # Extract the verified_result section
                    if "verified_result" in data:
                        verified_result = data["verified_result"]
                        thinking_eval = verified_result.get("thinking_eval")

                        # Calculate utility based on thinking_eval
                        utility = 0
                        if thinking_eval == 2:
                            utility = 1.0
                        elif thinking_eval == 1:
                            utility = 0.5
                        # thinking_eval = 0 corresponds to utility = 0

                        utility_values.append(utility)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing file {json_file}: {e}")

            # Calculate average utility score for this model in this setting
            if utility_values:
                avg_utility = sum(utility_values) / len(utility_values)
                all_results[setting][model] = {
                    "utility_score": avg_utility,
                    "sample_count": len(utility_values),
                }
            else:
                all_results[setting][model] = {"utility_score": 0, "sample_count": 0}

    return all_results


def save_results_to_json(
    results, output_path="./processed_results_temp1/utility_score.json"
):
    """
    Save the calculated results to a JSON file.

    Args:
        results: Dictionary containing the utility score results
        output_path: Path where the JSON file will be saved
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create absolute path relative to the script location
    absolute_output_path = os.path.join(script_dir, output_path)

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(absolute_output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save results as JSON
    with open(absolute_output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {absolute_output_path}")


def main():
    utility_scores = calculate_utility_scores(
        base_directory="./verified_results"
    )

    # Print results
    print("Utility Scores Across All Settings:")
    print("=" * 70)

    for setting, models in utility_scores.items():
        print(f"\nSetting: {setting}")
        print("-" * 50)
        print(f"{'Model':<30} {'Utility Score':<15} {'Sample Count':<15}")
        print("-" * 70)

        # Sort models by utility score for better readability
        sorted_models = sorted(
            models.items(), key=lambda x: x[1]["utility_score"], reverse=True
        )

        for model, data in sorted_models:
            score = data["utility_score"]
            count = data["sample_count"]
            print(f"{model:<30} {score:.4f}{' '*10} {count}")

        # Find best model for this setting
        if models:
            best_model = max(models.items(), key=lambda x: x[1]["utility_score"])
            print(
                f"\nBest model for {setting}: {best_model[0]} with score {best_model[1]['utility_score']:.4f}"
            )

    # Save results to JSON file
    save_results_to_json(utility_scores, output_path="../processed_results/utility_score.json")


if __name__ == "__main__":
    main()
