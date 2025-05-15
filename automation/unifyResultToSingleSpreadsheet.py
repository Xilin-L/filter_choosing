import os
import pandas as pd
import json

# Directory containing the JSON files
results_dir = '/home/xilin/projects/recon_ws/EfficientScans/results'
# Output Excel file
output_excel = '/home/xilin/projects/recon_ws/EfficientScans/classifiedResults.xlsx'

# List to store data from all JSON files
data_list = []

# Iterate over all JSON files in the results directory
for json_file in os.listdir(results_dir):
    if json_file.endswith('_result.json'):
        json_path = os.path.join(results_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Extract sampleMaterial, sampleDiameterMm, kvp, and binning from the filename
            filename_parts = json_file.split('_')
            sampleMaterial = filename_parts[0]
            sampleDiameterMm = filename_parts[1].replace('mm', '')
            kvp = filename_parts[2].replace('kv', '')
            binning = filename_parts[3].replace('bin', '')
            data['sampleMaterial'] = sampleMaterial
            data['sampleDiameterMm'] = sampleDiameterMm
            data['kvp'] = kvp
            data['binning'] = binning
            # separate BHC and scattering results
            bhc = data.pop('BHC', [None, None, None])
            scattering = data.pop('Scattering', [None, None])
            data['BHC'] = bhc[0]
            data['BHC_Simu'] = bhc[1]
            data['BHC_Theo'] = bhc[2]
            data['Scattering'] = scattering[0]
            data['Scattering_Esti'] = scattering[1]
            # Separate SNR and resolution results
            snr = data.pop('SNR', [None, None, None])
            reso = data.pop('Resolution', [None, None, None])
            reso_px = data.pop('ResolutionPx', [None, None, None])

            data['SNR_X'] = snr[0]
            data['SNR_Y'] = snr[1]
            data['SNR_Z'] = snr[2]
            data['Resolution_X'] = reso[0]
            data['Resolution_Y'] = reso[1]
            data['Resolution_Z'] = reso[2]
            data['ResolutionPx_X'] = reso_px[0]
            data['ResolutionPx_Y'] = reso_px[1]
            data['ResolutionPx_Z'] = reso_px[2]
            data_list.append(data)

# Create a DataFrame from the list of data
df = pd.DataFrame(data_list)

# Round the DataFrame values to 4 significant figures
df = df.round(4)

# Reorder the columns
df = df[['sampleMaterial', 'sampleDiameterMm', 'kvp', 'BHC', 'BHC_Simu', 'BHC_Theo', 'Scattering', 'Scattering_Esti',
         'SNR_X', 'SNR_Y', 'SNR_Z', 'Resolution_X', 'Resolution_Y', 'Resolution_Z', 'ResolutionPx_X', 'ResolutionPx_Y',
            'ResolutionPx_Z', 'binning']]
# Write the DataFrame to an Excel file, classifying by sampleMaterial and binning
with pd.ExcelWriter(output_excel) as writer:
    for (sampleMaterial, binning), group in df.groupby(['sampleMaterial', 'binning']):
        # Sort each group by sampleDiameterMm and kvp
        group = group.sort_values(by=['sampleDiameterMm', 'kvp'])
        sheet_name = f"{sampleMaterial}_bin{binning}"
        group.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Data has been written to {output_excel}")