import re
import argparse
import os
import numpy as np
import glob
import netCDF4 as nc

def extractMetadata(directoryPath):
    """Extract metadata from the specified file in the directory and its subdirectories."""
    keywords = ["diameter", "source_filter_thickness_mm", "voxel_size", "x_ray_energy", "subsample_factor"]
    results = {keyword: None for keyword in keywords}

    # Search for the file 'expt.in' in the directory and subdirectories
    filePath = glob.glob(os.path.join(directoryPath, '**', 'expt.in'), recursive=True)
    if not filePath:
        print("No 'expt.in' file found.")
        return None, None, None, None

    with open(filePath[0], 'r') as file:
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
    binning = results.get("subsample_factor")

    return sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp, binning

def processFiles(directoryPath, pattern, shape, dtype=np.uint16):
    """Process files matching the pattern and return their average as a NumPy array."""
    files = glob.glob(os.path.join(directoryPath, '**', pattern), recursive=True)
    data = []

    for file in files:
        array = np.fromfile(file, dtype=dtype).reshape(shape)
        data.append(array)

    if data:
        return np.mean(data, axis=0)
    else:
        print("No files found.")
        return None


def findMiddleKFFile(directoryPath, shape,dtype=np.uint16):
    """Find the file containing the pattern 'KF' and take only the middle one if sorted by their name."""
    files = sorted(glob.glob(os.path.join(directoryPath, '**', '*KF*.raw'), recursive=True))

    if not files:
        print("No KF files found.")
        return None

    middle_index = len(files) // 2
    kfMiddle = np.fromfile(files[middle_index], dtype=dtype).reshape(shape)
    return kfMiddle


def extractAllData(directoryPath,shape, offset=10000):
    """
    Extract metadata from expt.in, obtain the middle KF projection and average DF and CF data.
    :param directoryPath: the path that contains all the files to be processed
    :return: sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp, dfAverage, cfAverage,kfMiddle
    """

    sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp ,binning= extractMetadata(directoryPath)


    dfAverage = processFiles(directoryPath, '*DF*.raw', shape=shape)
    cfAverage = processFiles(directoryPath, '*CF*.raw', shape=shape)

    kfMiddle = findMiddleKFFile(directoryPath,shape=shape)

    tomoFilePath = glob.glob(os.path.join(directoryPath, '**', 'tomoLoRes.nc'), recursive=True)
    if not tomoFilePath:
        print("No 'tomoLoRes.nc' file found.")
        tomoLoRes = None
    else:
        tomoData = np.nan_to_num(
            np.array(nc.Dataset(tomoFilePath[0]).variables['tomo'][:], dtype=np.float32, copy=True), nan=0.0)
        tomoLoRes = np.copy(tomoData) - offset  # shift back the values
        tomoLoRes[tomoLoRes < 0] = 0  # Set negative values to 0


    return sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp, binning, dfAverage, cfAverage,kfMiddle, tomoLoRes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract metadata from a file and process DF and CF files.')
    parser.add_argument('directoryPath', type=str, help='Path to the directory containing the files')
    args = parser.parse_args()

    results = extractAllData(args.directoryPath)
    sampleDiameterMm, filterThicknessMm, vxSizeMm, kvp, binning, dfAverage, cfAverage, kfMiddle, tomoLoRes = results

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

    if kfMiddle:
        print(f"Middle KF file: {kfMiddle}")
    else:
        print("No KF file found.")