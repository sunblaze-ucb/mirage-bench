import os

for filename in os.listdir('/Users/pujiayue/Documents/research/llm agent hallucination/llm_agent_hallucination_data/dataset_all/unachievable/webarena'):
    if filename.endswith('.json') and 'unreachable' in filename:
        new_name = filename.replace('unreachable', 'unachievable')
        os.rename(filename, new_name)
        print(f"Renamed: {filename} -> {new_name}")
