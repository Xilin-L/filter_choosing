import pandas as pd

def process_csv(input_file, output_file):
    # Load the CSV file and drop any rows where critical columns are missing
    df = pd.read_csv(input_file).dropna(subset=['Experiment'])

    # Update the regular expression to capture `sstl` as both decimal and integer values
    extracted_columns = df['Experiment'].str.extract(r'(\d+mm_FeO)_(\d+)(kV)_(\d+(?:pt\d+)?|pt\d+)sstl(?:_(ANU\d)?)?')
    extracted_columns[4] = extracted_columns[4].fillna('ANU3')  # Default missing machine to 'ANU3'
    extracted_columns.columns = ['Base', 'kVp', 'kVp_unit', 'sstl', 'machine']

    # Add extracted columns to the dataframe
    df[['Base', 'kVp', 'sstl', 'machine']] = extracted_columns[['Base', 'kVp', 'sstl', 'machine']]

    # Check for rows where `kVp` is NaN and print them for debugging
    if df['kVp'].isna().any():
        print("Rows with NaN in `kVp` after extraction:")
        print(df[df['kVp'].isna()])

    # Drop rows where `kVp` is NaN after extraction (indicating an incomplete match)
    df = df.dropna(subset=['kVp'])

    # Convert kVp to integer since we know all values are valid
    df['kVp'] = df['kVp'].astype(int)

    # Convert sstl thickness to float by replacing 'pt' with '.' in thickness values
    df['sstl thickness'] = df['sstl'].str.replace('pt', '.').astype(float)

    # Pivot the table to get SNR values for each Slice Dimension
    pivot_df = df.pivot_table(index=['Base', 'machine', 'kVp', 'sstl thickness'], columns='Slice Dimension', values='SNR').reset_index()

    # Rename columns to match the desired output format
    pivot_df.columns.name = None  # Remove index name
    pivot_df.columns = ['Experiment', 'Machine', 'kVp', 'sstl thickness', 'SNR X', 'SNR Y', 'SNR Z']

    # Sort the pivoted DataFrame by 'Experiment', 'kVp', 'sstl thickness', and 'Machine'
    pivot_df = pivot_df.sort_values(by=['Experiment', 'kVp', 'sstl thickness', 'Machine']).reset_index(drop=True)

    # Save the sorted DataFrame to CSV
    pivot_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = '/home/xilin/projects/Testing123/snr_results_new.csv'  # Replace with your input file name
    output_file = '/home/xilin/projects/Testing123/snr_results_new_reformed.csv'  # Replace with your output file name
    process_csv(input_file, output_file)
