import os
import json
import glob
from collections import defaultdict
from pathlib import Path


def calculate_hallucination_rates(
    base_directory="llm_agent_hallucination_data/verified_results",
):
    """
    Calculate hallucination rates for all models across all settings.

    Hallucination Rate = (Number of samples with Utility=0) / (Total number of samples)

    Args:
        base_directory: Base path to the verified results directory

    Returns:
        Dictionary mapping settings to model hallucination rates
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

            total_samples = 0
            hallucination_count = 0

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

                        # Increment total samples
                        total_samples += 1

                        # Increment hallucination count if thinking_eval is 0
                        if thinking_eval == 0:
                            hallucination_count += 1

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing file {json_file}: {e}")

            # Calculate hallucination rate for this model in this setting
            if total_samples > 0:
                hallucination_rate = hallucination_count / total_samples
                all_results[setting][model] = {
                    "hallucination_rate": hallucination_rate,
                    "hallucination_count": hallucination_count,
                    "sample_count": total_samples,
                }
            else:
                all_results[setting][model] = {
                    "hallucination_rate": 0,
                    "hallucination_count": 0,
                    "sample_count": 0,
                }

    return all_results


def save_results_to_json(
    results, output_path="./processed_results/hallucination_rate.json"
):
    """
    Save the calculated results to a JSON file.

    Args:
        results: Dictionary containing the hallucination rate results
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
    hallucination_rates = calculate_hallucination_rates(
        base_directory="./verified_results"
    )

    # Print results
    print("Hallucination Rates Across All Settings:")
    print("=" * 80)

    for setting, models in hallucination_rates.items():
        print(f"\nSetting: {setting}")
        print("-" * 60)
        print(
            f"{'Model':<30} {'Hallucination Rate':<20} {'Hallucinations':<15} {'Sample Count':<15}"
        )
        print("-" * 80)

        # Sort models by hallucination rate for better readability (ascending order)
        sorted_models = sorted(models.items(), key=lambda x: x[1]["hallucination_rate"])

        for model, data in sorted_models:
            rate = data["hallucination_rate"]
            hall_count = data["hallucination_count"]
            total = data["sample_count"]
            print(f"{model:<30} {rate:.4f}{' '*15} {hall_count:<15} {total}")

        # Find best model (lowest hallucination rate) for this setting
        if models:
            best_model = min(models.items(), key=lambda x: x[1]["hallucination_rate"])
            print(
                f"\nBest model for {setting}: {best_model[0]} with hallucination rate {best_model[1]['hallucination_rate']:.4f}"
            )

    # Save results to JSON file
    save_results_to_json(hallucination_rates, output_path="../processed_results/hallucination_rate.json")


if __name__ == "__main__":
    main()
