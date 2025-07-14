import os
import pandas as pd
from collections import Counter

folder_path = r"E:\VitMix\student_masks"
output_file = r"E:\VitMix\student_masks\combined.csv"

if os.path.exists(output_file):
    try:
        os.remove(output_file)
        print(f"Existing file '{output_file}' has been deleted.")
    except Exception as e:
        print(f"Error deleting file '{output_file}': {e}")
        exit(1)

dataframes = []
total_files = 0
empty_files = 0
error_files = 0
initial_row_count = 0  # Rows before combining
processed_row_count = 0  # Rows after combining

# Recursively search for CSV files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', engine='python')

                if df.empty:
                    empty_files += 1
                    print(f"Skipping empty file: {file_path}")
                    continue

                row_count = len(df)
                initial_row_count += row_count
                dataframes.append(df)
                print(f"Loaded: {file_path} with {row_count} rows.")
            except Exception as e:
                error_files += 1
                print(f"Error reading {file_path}: {e}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    processed_row_count = len(combined_df)

    combined_df.to_csv(output_file, index=False)
    print(f"\nAll CSV files in {folder_path} and subfolders have been combined into {output_file}.")

    if 'filename' in combined_df.columns:
        filename_counts = combined_df['filename'].value_counts()
        unique_filenames = filename_counts.count()
        duplicates_in_dataset = (filename_counts > 1).sum()
        max_occurrences = filename_counts.max()

        print("\nAnalysis of 'filename' Column:")
        print(f"Total Unique Filenames: {unique_filenames}")
        print(f"Duplicate Filenames in Dataset: {duplicates_in_dataset}")
        print(f"Max Occurrences of a Single Filename: {max_occurrences}")
    else:
        print("\nWarning: 'filename' column not found in the combined dataset.")
else:
    print("No valid CSV files found.")

print("\nOverall Statistics:")
print(f"Total CSV files found: {total_files}")
print(f"Total valid files processed: {total_files - empty_files - error_files}")
print(f"Total empty files skipped: {empty_files}")
print(f"Total files with errors: {error_files}")
print(f"Total rows in all initial files: {initial_row_count}")
print(f"Total rows in combined file: {processed_row_count}")

# Validate row counts
if initial_row_count == processed_row_count:
    print("Row count matches! No data was missed.")
else:
    print(f"Row count mismatch! Initial: {initial_row_count}, Combined: {processed_row_count}.")
