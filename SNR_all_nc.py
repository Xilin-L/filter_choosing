import os
import csv
import re
import numpy as np
from snr_test import tomoSliceSNR

def process_nc_files(base_path):
    results = []
    pattern = re.compile(r'tomoSlice([XYZ])')

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".nc"):
                match = pattern.search(file)
                if match:
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(root)
                    snr = tomoSliceSNR(file_path)
                    new_file_name = match.group(1)
                    results.append([folder_name, new_file_name, snr])

    return results

def save_results_to_csv(results, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "Slice Dimension", "SNR"])
        writer.writerows(results)

if __name__ == "__main__":
    base_path = '/home/xilin/projects/Testing123/'
    output_csv = '/home/xilin/projects/Testing123/snr_results.csv'
    results = process_nc_files(base_path)
    save_results_to_csv(results, output_csv)
    print(f"Results saved to {output_csv}")