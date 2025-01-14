import os
import csv
import re
import numpy as np
from snrTest import tomoSliceSNR

def process_nc_files(base_path, output_csv):
    pattern = re.compile(r'tomoSlice([XYZ])')

    # Open the CSV file in append mode and write the header if it doesn’t exist
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Check if the file is empty to add headers only once
        if file.tell() == 0:
            writer.writerow(["Experiment", "Slice Dimension", "SNR"])

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".nc"):
                match = pattern.search(file)
                if match:
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(root)

                    # Calculate SNR
                    snr = tomoSliceSNR(file_path)
                    slice_dimension = match.group(1)  # Extract X, Y, or Z from filename

                    # Write the result to the CSV file
                    with open(output_csv, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([folder_name, slice_dimension, snr])
                        print(f"Updated {output_csv} with {file_path} SNR.")


if __name__ == "__main__":
    base_path = '/home/xilin/projects/Testing123/'
    output_csv = '/home/xilin/projects/Testing123/snr_results_new2.csv'

    # Process files and update results in CSV after each calculation
    process_nc_files(base_path, output_csv)
    print(f"All results updated in {output_csv}")