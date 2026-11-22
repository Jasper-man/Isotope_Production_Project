import curie as ci
import numpy as np
import pandas as pd

sp = ci.Spectrum('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Raw Spectra/25 MeV Protons/FP1/20260830_Det2_FP1_50cm_job000.Spe')
sp.cb = '/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Curie_Calibrations/Det2/Det2_cb_50.json'
sp.isotopes = ['196AU']
sp.fit_peaks(fixed_energy = False)
print(type(sp.peaks))
counts = sp.peaks.iloc[0]['counts']
unc_counts = sp.peaks.iloc[0]['unc_counts']
chi2 = sp.peaks.iloc[0]['chi2']
timestamp = sp.peaks.iloc[0]['start_time']
live_time = sp.peaks.iloc[0]['live_time']

print(counts)
print(unc_counts)
print(chi2)
print(timestamp)
print(live_time)


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', 1000)
# print(sp.peaks)


# def get_counts_by_energy(sp, target_energy: float, tolerance: float = 2.0):
#     """
#     Finds the peak closest to the target energy and returns its counts.
#     'tolerance' ensures we don't accidentally grab a peak 50 keV away if yours is missing.
#     """
#     # 1. Safety check: Did curie find any peaks at all?
#     if sp.peaks is None or sp.peaks.empty:
#         return None
        
#     # 2. Calculate how far every peak in the table is from your target
#     # .abs() makes sure we only deal with positive differences
#     energy_differences = (sp.peaks['energy'] - target_energy).abs()
    
#     # 3. Find the row index with the smallest difference
#     closest_idx = energy_differences.idxmin()
    
#     # 4. Check if the closest peak is actually within your tolerance window (e.g., +/- 2 keV)
#     if energy_differences[closest_idx] <= tolerance:
#         # Extract the counts from that specific row
#         counts = sp.peaks.loc[closest_idx, 'counts']
#         return int(counts)
#     else:
#         print(f"No peak found within {tolerance} keV of {target_energy}.")
#         return None

# # counts = get_counts_by_energy(sp, 1293)
# # print(counts)

# sp.isotopes = ['41AR']
# counts = get_counts_by

# # 2. The Diagnostic Block
# print("\n--- DIAGNOSTIC CHECK ---")
# print(f"File being checked: {sp.filename}")

# if sp.peaks is None:
#     print("Result: sp.peaks is completely None. The fit failed or found nothing.")
# else:
#     print(f"Is the table empty? {sp.peaks.empty}")
#     print("\nHere is the raw table:")
    
#     # This prints all rows and columns to your terminal
#     print(sp.peaks) 
    
#     # If you want to see all the column names you can use:
#     # print("\nAvailable columns:", sp.peaks.columns.tolist())
    
#     # Optional: Save this single result to a CSV to look at in Excel
#     # sp.peaks.to_csv("test_peaks_output.csv", index=False)
    
# print("------------------------\n")

