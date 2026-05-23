import json
import csv
import os

def convert_jsonl_to_csv(jsonl_file, csv_file):
    """Convert JSONL file to CSV format"""
    data = []

    # Read JSONL file
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    if not data:
        print("No data found in JSONL file")
        return

    # Get all unique keys from the data
    fieldnames = set()
    for item in data:
        fieldnames.update(item.keys())

    fieldnames = sorted(list(fieldnames))  # Sort for consistent column order

    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Converted {len(data)} records from {jsonl_file} to {csv_file}")
    print(f"Columns: {', '.join(fieldnames)}")

if __name__ == "__main__":
    jsonl_file = "dataset_ormas/ormas_liputan6.jsonl"
    csv_file = "dataset_ormas/ormas_liputan6.csv"

    if os.path.exists(jsonl_file):
        convert_jsonl_to_csv(jsonl_file, csv_file)
    else:
        print(f"File {jsonl_file} not found")
