import re
import argparse
import os
import numpy as np
import glob

def extractMetadata(filePath):
    """Extract metadata from the specified file."""
    keywords = ["diameter", "source_filter_thickness_mm", "voxel_size", "x_ray_energy"]
    results = {keyword: None for keyword in keywords}

    with open(filePath, 'r') as file:
        for line in file:
            for keyword in keywords:
                if keyword in line:
                    match = re.search(rf"{keyword}\s*[:=]?\s*(\d+\.?\d*)", line)
                    if match:
                        results[keyword] = float(match.group(1))
                        break

    sampleDiameterMm = results.get("diameter")
    filterThicknessMm = results.get("source_filter_thickness_mm")
    vxSizeMm = results.get("voxel_size") / 1000
    kvp = results.get("x_ray_energy")

    return sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp

def processFiles(directoryPath, pattern, dtype=np.uint16, shape=(1420, 1436)):
    """Process files matching the pattern and return their average as a NumPy array."""
    files = glob.glob(os.path.join(directoryPath, pattern))
    data = []

    for file in files:
        array = np.fromfile(file, dtype=dtype).reshape(shape)
        data.append(array)

    if data:
        return np.mean(data, axis=0)
    else:
        print("No files found.")
        return None

def main(directoryPath):
    fileName = 'expt.in'
    filePath = os.path.join(directoryPath, fileName)

    sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp = extractMetadata(filePath)

    dfAverage = processFiles(directoryPath, '*DF*.raw')
    cfAverage = processFiles(directoryPath, '*CF*.raw')

    return sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp, dfAverage, cfAverage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract metadata from a file and process DF and CF files.')
    parser.add_argument('directoryPath', type=str, help='Path to the directory containing the files')
    args = parser.parse_args()

    results = main(args.directoryPath)
    sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp, dfAverage, cfAverage = results

    print(f"sampleDiameterMm: {sampleDiameterMm}")
    print(f"filterThicknessMm: {filterThicknessMm}")
    print(f"vxSizeMm: {vxSizeMm}")
    print(f"kvp: {kvp}")

    if dfAverage is not None:
        print("Average dark field values obtained.")
    else:
        print("No DF files found.")

    if cfAverage is not None:
        print("Average clear field values obtained.")
    else:
        print("No CF files found.")