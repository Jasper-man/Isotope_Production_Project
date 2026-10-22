import curie as ci
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm
import re
import csv

# home-made functions
from abnormal_shape_SA import squash_beam
from decay_function import decay_function
# from PythonPrograms.Serber_Proton_Yield_Calculations import serber_yield


"""
I want to build a csv for each run that will have the name of each file, the associated distance, the calibrattion file that matches, the isotope we are looking for
the counts of that isotope, the time that the file was running for and the uncertanties, I will be building this in pieces

# I want to make a function that will go through a file and apply a curie analysis to the spectra, looking for a particular gamma and determining the activity

# I have a lot of files to go through for the foil pack so I want to go through the names and grab the distances so I can automate the curie analysis and apply
# the appropriate calibration

"""


# This is a helper function that will go through the spectra files and pull out the distances
def extract_distance(filename: str) -> str | None:
    """Extracts numeric distance following 'FPI_' from a filename."""
    # This pattern defines a group to be identified via the inner group following FP1 anf before cm
    pattern = r"FP1_(\d+(?:\.\d+)?)(?:cm)?"
    # clarify that the program should ignore case
    match = re.search(pattern, filename, re.IGNORECASE)
    return match.group(1) if match else None

# This function will build the dataset which is passed between the different functions and added to by initializing rows to be filled later and
# filling the name and distance row
def build_initial_dataset(spectra_folder: str) -> list[dict]:
    # assign the folder th the path associated with the stored spectra
    folder = Path(spectra_folder)

    # throw an error if there is an incorrect path or the path is not a folder
    if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder}")
    

    dataset = []

    for item in sorted(folder.iterdir()):
        if item.is_file():
            distance = extract_distance(item.name)
            
            # Create a row for your future CSV
            row = {
                "File_Name": item.name,
                "Distance_cm": distance,
                "Calibration_File": None,  # To be filled by calibration function
                "Isotope": None,           # to be filled by the counting function
                "Chi2": None,
                "Gamma_Counts": None,      # To be filled by counting function
                "Energy": None,             # To be filled by counting function
                "Uncertainty": None,        # To be filled by counting function
                "Run_Time_s": None,        # To be filled by counting function
                "Timestamp_s": None       # To be filled by the counting function
            }
            dataset.append(row)
            
    return dataset

# using the distances found by the previous function this file will determine the distance that will match each file in the folder and assign it to
# the spectra which the curie program will use
from pathlib import Path

def match_calibration_files(dataset: list[dict], calib_folder: str) -> list[dict]:
    calib_path = Path(calib_folder)
    
    if not calib_path.exists():
        print(f"Error: Calibration folder '{calib_folder}' not found.")
        return dataset

    for row in dataset:
        distance = row["Distance_cm"]
        
        if not distance:
            continue
            
        matched_cal_file = None
        for cal_file in calib_path.iterdir():
            if not cal_file.is_file():
                continue
                
            # Look specifically for files ending with "18.json", "5.5.json", etc.
            # .lower() ensures it matches ".JSON" or ".json" safely
            target_suffix = f"{distance}.json".lower()
            
            if cal_file.name.lower().endswith(target_suffix):
                matched_cal_file = cal_file.name
                break 
        
        if matched_cal_file:
            row["Calibration_File"] = matched_cal_file
        else:
            print(f"Warning: No calibration file found ending in {distance}.json (Spectrum: {row['File_Name']})")
            
    return dataset

# diagnosis function for matching the data to a calib file
def diagnose_calibration_match(dataset: list[dict], calib_folder: str):
    calib_path = Path(calib_folder)
    
    print("\n--- STARTING DIAGNOSTIC RUN ---")
    
    for row in dataset:
        distance = row["Distance_cm"]
        if not distance:
            continue
            
        print(f"\nLooking for distance: {repr(distance)}")
        
        for cal_file in calib_path.iterdir():
            if not cal_file.is_file():
                continue
                
            target_suffix = f"{distance}.json".lower()
            actual_name = cal_file.name.lower()
            
            # This prints the exact comparison happening in the background
            print(f"  Checking file: {repr(actual_name)} | Does it end with {repr(target_suffix)}?")
            
            if actual_name.endswith(target_suffix):
                print(f"  >>> SUCCESS: Matched {cal_file.name}")
                break 

    print("--- END DIAGNOSTIC RUN ---\n")
    return dataset



# The curie part of the analysis that records the timestamps, livetimes, counts and uncertainty for a desired isotope
def calculate_gamma_counts(dataset: list[dict], 
                           spectra_folder: str, 
                           calib_folder: str, 
                           isotope: str, 
                           target_energy: float = None, 
                           energy_tol: float = 1.0) -> list[dict]:
    """
    Uses the curie library to calculate counts for a specific isotope.
    Updates the dataset in memory.
    """
    spectra_path = Path(spectra_folder)
    calib_path = Path(calib_folder)

    for row in tqdm(dataset):
        spec_file = row["File_Name"]
        calib_file = row["Calibration_File"]

        # Skip if this row doesn't have a matched calibration file
        if not calib_file:
            print(f"Skipping {spec_file}: No calibration file.")
            continue
            
        # Build the full system paths so curie can find them
        full_spec_path = str(spectra_path / spec_file)
        full_calib_path = str(calib_path / calib_file)

        try:
            # 1. Load the Calibration and Spectrum using curie
            # (Note: Check Curie's API for the exact argument to attach the calibration)
            sp = ci.Spectrum(full_spec_path)
            sp.cb = full_calib_path
            sp.isotopes = [isotope]
            
            # Adjustments to the peak fitting, lowering the peak to bg threshold
            sp.fit_config = {'SNR_min':2}
            
            # 2. Tell curie to fit the peaks for your target isotope
            # (Adjust to match Curie's specific fitting method, e.g., sp.fit_peaks())
            sp.fit_peaks() 
            
            # quick check to see if the peaks we are looking for exsist above SNR = 2
            if sp.peaks is None or sp.peaks.empty:
                print(f"Warning: No peaks fit for {isotope} in {spec_file}.")
                continue
              
            # Isolate The specific gamma
            if target_energy is not None:
                matched_peaks = sp.peaks[np.isclose(sp.peaks['energy'], target_energy, atol=energy_tol)]
                if matched_peaks.empty:
                    print(
                        f"Warning: Target energy {target_energy} keV not found in {spec_file}. "
                        f"Fitted energies: {sp.peaks['energy'].tolist()}"
                    )
                    continue
                # If multiple lines are within tolerance, pick the closest one
                closest_idx = (matched_peaks['energy'] - target_energy).abs().idxmin()
                peak_row = matched_peaks.loc[closest_idx]
            else:
                peak_row = sp.peaks.iloc[0]

            # 3. Extract the data you need from the curie Spectrum object
            # Replace these properties with curie's exact attribute names

            # 4. Inject the results DIRECTLY back into the dictionary
            row["Gamma_Counts"] = peak_row['counts']
            row["Energy"] = peak_row['energy']
            row["Isotope"] = peak_row['isotope']
            row["Run_Time_s"] = peak_row['live_time']
            row["Uncertainty"] = peak_row['unc_counts']
            row["Timestamp_s"] = peak_row['start_time']
            row["Chi2"] = peak_row['chi2']
            
            print(f"Success: {spec_file} -> completed")

        except Exception as e:
            # If curie fails (e.g., peak not found, file corrupted), catch the error
            print(f"Error analyzing {spec_file}: {e}")

    return dataset

# The fial step of the function that will turn the dataset into a csv, which can be moved to excel or another visualization software
def save_to_csv(dataset: list[dict], output_filename: str):
    if not dataset:
        print("Dataset is empty, nothing to save.")
        return

    # 1. Grab the column headers from the keys of the first dictionary
    fieldnames = dataset[0].keys()

    # 2. Open the file in 'w' (write) mode to initialize it
    with open(output_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # 3. This single line initializes the CSV with your column headers
        writer.writeheader()
        
        # 4. Write all the populated data row by row
        for row in dataset:
            writer.writerow(row)
            
    print(f"Successfully saved {len(dataset)} rows to {output_filename}")


# The function that puts all of the components together into one clean workflow
def Data_Analysis(spectra_folder: str, calib_folder: str, isotope: str, output_filename: str, target_energy: float = None, energy_tol: float = 1.0):

    dataset = build_initial_dataset(spectra_folder)
    # diagnose_calibration_match(dataset, calib_folder)
    match_calibration_files(dataset, calib_folder)
    calculate_gamma_counts(dataset,spectra_folder,calib_folder,isotope,target_energy,energy_tol)
    save_to_csv(dataset,output_filename)

    return "Data Analysis Complete"


spectra_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1'
calib_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2'
isotope = '196AU'
output_filename = '196Au_Counts_FP1_355keV.csv'

Data_Analysis(spectra_folder,calib_folder,isotope,output_filename,355)



spectra_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1'
calib_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2'
isotope = '116IN'
output_filename = '116In_Counts_FP1_463keV.csv'

# Data_Analysis(spectra_folder,calib_folder,isotope,output_filename,463.0)


spectra_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1'
calib_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2'
isotope = '57NI'
output_filename = '57Ni_Counts_FP1_127keV'

# Data_Analysis(spectra_folder,calib_folder,isotope,output_filename,127.164)


spectra_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1'
calib_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2'
isotope = '196AU'
output_filename = '116In_Counts_FP1'

# Data_Analysis(spectra_folder,calib_folder,isotope,output_filename)


spectra_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1'
calib_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2'
isotope = '89ZR'
output_filename = '89ZR_Counts_FP1'

# Data_Analysis(spectra_folder,calib_folder,isotope,output_filename)


spectra_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1'
calib_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2'
isotope = '116IN'
output_filename = '116In_Counts_FP1'

# Data_Analysis(spectra_folder,calib_folder,isotope,output_filename)

"""
This section will use the csv created to build the decay equation and plot the decay based on the timestamps in the data
"""

# 1. Define the mathematical decay model
def decay_equation(t, A0, decay_constant):
    """
    t: time elapsed since the reference point (in seconds)
    A0: Activity at t=0
    decay_constant: lambda (ln(2) / half-life)
    """
    return A0 * np.exp(-decay_constant * t)

def plot_and_fit_decay(csv_filepath: str, reference_time_str: str, target_distance: float = None):
    # 2. Load the data
    df = pd.read_csv(csv_filepath)
    
    # Optional: Filter for a specific distance if your CSV has multiple distances
    if target_distance:
        df = df[df["Distance_cm"] == target_distance]
        if df.empty:
            print(f"No data found for distance {target_distance}cm")
            return

    # 3. Convert times to usable numbers
    # Parse the strings into actual datetime objects
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    ref_time = pd.to_datetime(reference_time_str)
    
    # Calculate exactly how many seconds passed between your reference time and the start of each count
    df['Elapsed_Seconds'] = (df['Start_Time'] - ref_time).dt.total_seconds()
    
    # 4. Calculate Count Rates (Activity)
    # You must divide by Live_Time_s, otherwise longer runs artificially look like higher activity
    df['Count_Rate'] = df['Gamma_Counts'] / df['Live_Time_s']
    df['Rate_Uncertainty'] = df['Uncertainty'] / df['Live_Time_s']
    
    # Drop any rows where curie failed to find a peak (NaN values)
    df = df.dropna(subset=['Count_Rate', 'Elapsed_Seconds'])
    
    # Extract as standard numpy arrays for the math functions
    t_data = df['Elapsed_Seconds'].values
    y_data = df['Count_Rate'].values
    y_err = df['Rate_Uncertainty'].values
    
    # 5. Fit the Curve
    # p0 is our initial guess to help the algorithm: [guess_for_A0, guess_for_lambda]
    initial_guess = [y_data.max(), 1e-4]
    
    try:
        # curve_fit returns the optimal parameters (popt) and the covariance/error matrix (pcov)
        popt, pcov = curve_fit(
            decay_equation, 
            t_data, 
            y_data, 
            sigma=y_err,         # Weigh the fit using your uncertainties
            absolute_sigma=True, # Treats the uncertainties as real physical errors, not relative weights
            p0=initial_guess
        )
        
        A0_fit, lambda_fit = popt
        A0_err, lambda_err = np.sqrt(np.diag(pcov)) # Extract the +/- error of the fit
        
        half_life_seconds = np.log(2) / lambda_fit
        
        print("\n--- FIT RESULTS ---")
        print(f"Reference Time (t=0): {ref_time}")
        print(f"Calculated Activity at t=0 (A0): {A0_fit:.2f} +/- {A0_err:.2f} counts/sec")
        print(f"Calculated Half-Life: {half_life_seconds / 3600:.2f} hours")
        print("-------------------\n")
        
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        return

    # 6. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot the raw data with error bars
    plt.errorbar(t_data, y_data, yerr=y_err, fmt='o', label='Measured Data', color='black', capsize=5)
    
    # Generate smooth time points from t=0 up to your last measurement
    t_smooth = np.linspace(0, t_data.max(), 500)
    
    # Plot the fitted curve
    plt.plot(t_smooth, decay_equation(t_smooth, A0_fit, lambda_fit), 
             label=f'Fit: $A_0$ = {A0_fit:.1f}\n$T_{{1/2}}$ = {half_life_seconds/3600:.1f} hrs', 
             color='red', linestyle='--')
    
    # Formatting
    plt.axvline(x=0, color='gray', linestyle=':', label='Reference Time (t=0)')
    plt.xlabel('Seconds since Reference Time')
    plt.ylabel('Count Rate (Counts / Live Time)')
    plt.title('Isotope Decay Curve Extrapolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

# --- How to run it ---
# if __name__ == "__main__":
#     csv_file = "60Co_analysis_results.csv" 
    
#     # Put the exact date/time you want to back-calculate to. 
#     # Use standard ISO format: YYYY-MM-DD HH:MM:SS
#     my_reference_time = "2026-09-03 08:00:00"
    
#     plot_and_fit_decay(csv_file, my_reference_time, target_distance=18.0)
