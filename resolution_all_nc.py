import os
import csv
import re
import numpy as np
from resolution_estimation import tomoSliceRes  # Assuming findImageRes is imported from resolution_estimation module


def process_nc_files(base_path, output_csv, pxSzMm=1.0, Ng=8, geometricMax=False, plot=True, crop=False):
    pattern = re.compile(r'tomoSlice([XYZ])')  # Pattern to identify slice dimension in filenames

    # Open the CSV file in append mode and write the header if it doesn’t exist
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Check if the file is empty to add headers only once
        if file.tell() == 0:
            writer.writerow(["Experiment", "Slice Dimension", "Resolution"])

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith(".nc"):
                    match = pattern.search(file)
                    if match:
                        file_path = os.path.join(root, file)
                        folder_name = os.path.basename(root)

                        # Calculate resolution
                        resolution = tomoSliceRes(file_path, pxSzMm, Ng, geometricMax, plot, crop)
                        slice_dimension = match.group(1)  # Extract X, Y, or Z from filename

                        # Open the CSV file in append mode and write the result
                        with open(output_csv, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([folder_name, slice_dimension, resolution])
                            print(f"Updated {output_csv} with {file_path} resolution.")




if __name__ == "__main__":

    base_path = '/home/xilin/projects/Testing123'  # Base directory to search
    output_csv = '/home/xilin/projects/Testing123/resolution_results.csv'

    # Process files and update results in CSV after each calculation
    process_nc_files(base_path, output_csv)
    print(f"All results updated in {output_csv}")
