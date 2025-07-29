#!/usr/bin/env python3
"""
Token counting script
Recursively traverse all JSON files in the specified directory, extract the 'input' field, and count the number of tokens.
"""

import os
import json
import tiktoken
from collections import defaultdict
import sys

def count_tokens(text, model="gpt-4o"):
    """Count the number of tokens in the text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def extract_input_content(input_data):
    """Extract all text content from the input data."""
    if not isinstance(input_data, list):
        return ""
    
    content_parts = []
    for item in input_data:
        if isinstance(item, dict) and 'content' in item:
            content_parts.append(str(item['content']))
    
    return ' '.join(content_parts)

def process_json_file(file_path):
    """Process a single JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'input' not in data:
            return None, f"Field 'input' not found in file {file_path}"
        
        input_content = extract_input_content(data['input'])
        if not input_content:
            return None, f"The 'input' field in file {file_path} is empty or incorrectly formatted"
        
        token_count = count_tokens(input_content)
        return token_count, None
        
    except json.JSONDecodeError as e:
        return None, f"JSON decode error in {file_path}: {e}"
    except Exception as e:
        return None, f"Error processing file {file_path}: {e}"

def main():
    base_dir = "/home/weichenzhang/hallucination/mirage-bench/dataset_all"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        sys.exit(1)
    
    print(f"Start scanning directory: {base_dir}")
    print("=" * 60)
    
    total_files = 0
    total_tokens = 0
    successful_files = 0
    failed_files = 0
    stats_by_category = defaultdict(lambda: {'files': 0, 'tokens': 0})
    
    # Recursively traverse all files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                total_files += 1
                
                # Get category information (based on directory structure)
                category = os.path.relpath(root, base_dir).split(os.sep)[0]
                
                print(f"Processing file {total_files}: {os.path.relpath(file_path, base_dir)}")
                
                token_count, error = process_json_file(file_path)
                
                if token_count is not None:
                    successful_files += 1
                    total_tokens += token_count
                    stats_by_category[category]['files'] += 1
                    stats_by_category[category]['tokens'] += token_count
                    print(f"  ✓ Token count: {token_count:,}")
                else:
                    failed_files += 1
                    print(f"  ✗ Error: {error}")
                
                print()
    
    # Output statistics
    print("=" * 60)
    print("Summary of statistics")
    print("=" * 60)
    print(f"Total files: {total_files:,}")
    print(f"Successfully processed: {successful_files:,}")
    print(f"Failed to process: {failed_files:,}")
    print(f"Total tokens: {total_tokens:,}")
    
    if successful_files > 0:
        print(f"Average tokens per file: {total_tokens / successful_files:,.1f}")
    
    print("\nStatistics by category:")
    print("-" * 40)
    for category, stats in sorted(stats_by_category.items()):
        if stats['files'] > 0:
            avg_tokens = stats['tokens'] / stats['files']
            print(f"{category:20s}: {stats['files']:4d} files, {stats['tokens']:10,} tokens (avg: {avg_tokens:6.1f})")
    
    print(f"\nStatistics completed! Processed {total_files} JSON files in total.")

if __name__ == "__main__":
    main() 