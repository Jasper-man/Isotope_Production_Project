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

# home-made functions
from abnormal_shape_SA import squash_beam
from decay_function import decay_function
from Serber_Yield_Calculations import serber_yield


"""
I want to build a csv for each run that will have the name of each file, the associated distance, the calibrattion file that matches, the isotope we are looking for
the counts of that isotope, the time that the file was running for and the uncertanties, I will be building this in pieces
"""


# starting with the calibration spectra
# sp_Ba133 = ci.Spectrum('/home/james_andrew/Python_Projects/Pb-212/Beam_Current_Calculations/Calibration_Spectra/Meng_133Ba_5.4cm_fixed.Spe')
# sp_Ba133.isotopes = ['133BA']
# sp_Eu152 = ci.Spectrum('/home/james_andrew/Python_Projects/Pb-212/Beam_Current_Calculations/Calibration_Spectra/Meng_152Eu_5.4cm_fixed.Spe')
# sp_Eu152.isotopes = ['152EU']

# cb_det2 = ci.Calibration()
# cb_det2.calibrate([sp_Ba133, sp_Eu152], sources=[{'isotope':'133BA', 'A0':3.859E4, 'ref_date':'03/01/2019 12:00:00'}, {'isotope':'152EU', 'A0':3.822E4, 'ref_date':'03/01/2019 12:00:00'}])
# cb_det2.saveas('Efficiency_Det2.json')

# I want to make a function that will go through a file and apply a curie analysis to the spectra, looking for a particular gamma and determining the activity

# I have a lot of files to go through for the foil pack so I want to go through the names and grab the distances so I can automate the curie analysis and apply
# the appropriate calibration

def extract_distance(filename: str) -> str | None:
    """Extracts numeric distance following 'FPI_' from a filename."""
    # This pattern defines a group to be identified via the inner group following FP1 anf before cm
    pattern = r"FP1_(\d+(?:\.\d+)?)(?:cm)?"
    # clarify that the program should ignore case
    match = re.search(pattern, filename, re.IGNORECASE)
    return match.group(1) if match else None

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
                "Gamma_Counts": None,      # To be filled by counting function
                "Run_Time_s": None,        # To be filled by time function
                "Uncertainty": None        # To be filled by stats function
            }
            dataset.append(row)
            
    return dataset




# def scan_and_extract_distances(folder_path: str | Path) -> dict[str, str]:
#     """
#     Iterates through a folder, extracts the file names one by one,
#     and feeds them into extract_distance.

#     Returns:
#         A dictionary mapping the filename to its extracted distance.
#     """
#     folder = Path(folder_path)

#     if not folder.exists():
#         raise FileNotFoundError(f"Folder not found: {folder}")
#     if not folder.is_dir():
#         raise NotADirectoryError(f"Path is not a directory: {folder}")

#     results = {}

#     # Iterate through each item inside the directory
#     for item in sorted(folder.iterdir()):
#         # Ensure we only process files, skipping subfolders
#         if item.is_file():
#             filename = item.name
#             print(filename)  # Extracts just the file name (e.g. 'run1_FPI_18cm.spe')
            
#             # Feed the filename into the extractor
#             distance = extract_distance(filename)

#             if distance is not None:
#                 results[filename] = distance
#                 print(f"File: {filename} -> Distance: {distance} cm")
#             else:
#                 print(f"File: {filename} -> No distance pattern found")

#     return results
# results = scan_and_extract_distances('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/FP1')
# print(type(results))
# print(results)