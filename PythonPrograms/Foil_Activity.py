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
from Serber_Yield_Calculations import serber_yield


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
                "Gamma_Counts": None,      # To be filled by counting function
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

# This is another helper function that will pull the counts out of the pandas data frame from sp.peaks and return the number and uncertainty
def get_isotope_counts(sp, target_isotope: str):
    # sp.peaks is a pandas DataFrame.
    # We filter it to only keep rows where the 'isotope' column matches your target.
    matched_rows = sp.peaks[sp.peaks['isotope'] == target_isotope]
    
    if matched_rows.empty:
        return None
        
    # Note: Some isotopes (like 60Co) emit multiple gamma rays, 
    # so they will have multiple peaks (rows) in this table.
    
    # If you just want the counts from the strongest/first peak found:
    raw_counts = matched_rows.iloc[0]['counts']
    
    # Or, if you need the uncertainty too:
    unc = matched_rows.iloc[0]['unc_counts']
    
    return int(raw_counts), int(unc)


# The curie part of the analysis that records the timestamps, livetimes, counts and uncertainty for a desired isotope
def calculate_gamma_counts(dataset: list[dict], spectra_folder: str, calib_folder: str, isotope: str) -> list[dict]:
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
            cb = ci.Calibration(full_calib_path)
            sp = ci.Spectrum(full_spec_path)
            
            # Attach the calibration to the spectrum
            sp.cb = cb 
            
            # 2. Tell curie to fit the peaks for your target isotope
            # (Adjust to match Curie's specific fitting method, e.g., sp.fit_peaks())
            sp.fit_peaks(isotope=isotope) 
            
            # 3. Extract the data you need from the curie Spectrum object
            # Replace these properties with curie's exact attribute names

            sp.isotopes = [isotope]
            # counts = sp.isotopes[isotope].counts
            # uncertainty = sp.isotopes[isotope].counts_unc
            run_time = sp.real_time
            timestamp = sp.start_time  

            # 4. Inject the results DIRECTLY back into the dictionary
            row["Gamma_Counts"] = counts
            row["Isotope"] = isotope
            row["Run_Time_s"] = run_time
            row["Uncertainty"] = uncertainty
            row["Timestamp_s"] = timestamp
            
            print(f"Success: {spec_file} -> {counts} counts for {isotope}")

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
def Data_Analysis(spectra_folder: str, calib_folder: str, isotope: str, output_filename: str):

    dataset = build_initial_dataset(spectra_folder)
    # diagnose_calibration_match(dataset, calib_folder)
    match_calibration_files(dataset, calib_folder)
    calculate_gamma_counts(dataset,spectra_folder,calib_folder,isotope)
    save_to_csv(dataset,output_filename)

    return "Data Analysis Complete"


spectra_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1'
calib_folder = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2'
isotope = '196AU'
output_filename = 'Data_Analysis_TEST'

Data_Analysis(spectra_folder,calib_folder,isotope,output_filename)


