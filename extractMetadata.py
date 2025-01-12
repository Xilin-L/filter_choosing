import re

# Define the file path and keywords to search
file_path = "/home/xilin/projects/recon_ws/EfficientScans/AL_33mm__180kV-80uA_bin2_450_CD1150mm/expt.in"
keywords = ["diameter", "source_filter_thickness_mm", "voxel_size","x_ray_energy"]

# Initialize a dictionary to store the numbers for each keyword
results = {keyword: None for keyword in keywords}

# Open the file and search for the keywords
with open(file_path, 'r') as file:
    for line in file:
        for keyword in keywords:
            if keyword in line:
                # Find the number next to the keyword
                match = re.search(rf"{keyword}\s*[:=]?\s*(\d+\.?\d*)", line)
                if match:
                    results[keyword] = float(match.group(1))  # Use int() if integers are expected
                    break  # Move to the next line after finding a match

# Assign the extracted numbers to variables
sampleDiameterMm = results.get("diameter")
filterThicknessMm = results.get("source_filter_thickness_mm")
vxSizeMm = results.get("voxel_size")/1000
kvp = results.get("x_ray_energy")

# Print the results
print(f"sampleDiameterMm: {sampleDiameterMm}")
print(f"filterThicknessMm: {filterThicknessMm}")
print(f"vxSizeMm: {vxSizeMm}")
print(f"kvp: {kvp}")
