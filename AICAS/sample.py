import json
import random
import argparse
from collections import defaultdict

# Define the target counts for each category
TARGET_COUNTS = {
    "Regular Text Recognition": 10,
    "Irregular Text Recognition": 10,
    "Artistic Text Recognition": 10,
    "Handwriting Recognition": 10,
    "Digit String Recognition": 10,
    "Non-Semantic Text Recognition": 10,
    "Scene Text-centric VQA": 40
}

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Sample items from a JSON file based on type.")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input JSON file (e.g., FullTest.json)"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path for the output sampled JSON file (e.g., sample_100.json)"
    )
    return parser.parse_args()

def main(args):
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            json_data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found: '{args.input}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{args.input}'")
        return
    except Exception as e:
        print(f"An unexpected error occurred while reading '{args.input}': {e}")
        return

    if not json_data_list:
        print("Input file is empty. No sampling performed.")
        return

    items_by_type = defaultdict(list)
    for item in json_data_list:
        if 'type' in item:
            items_by_type[item['type']].append(item)

    selected_items = []
    for type_name, required_count in TARGET_COUNTS.items():
        available_items = items_by_type.get(type_name, [])
        available_count = len(available_items)

        if available_count == 0:
            continue  # Skip if no items for this type

        # Determine the actual number to sample
        actual_count_to_sample = min(available_count, required_count)
        
        sampled_list = random.sample(available_items, actual_count_to_sample)
        selected_items.extend(sampled_list)

    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            # ensure_ascii=False to write CJK characters correctly
            # indent=4 for pretty printing
            json.dump(selected_items, outfile, ensure_ascii=False, indent=4)
        
        print(f"Successfully wrote {len(selected_items)} items to '{args.output}'")
    
    except IOError as e:
        print(f"Error: Failed to write to output file '{args.output}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing the file: {e}")


if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args)