import curie as ci
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from tqdm import tqdm
import re

# home-made functions
from abnormal_shape_SA import squash_beam
from decay_function import decay_function
from Serber_Yield_Calculations import serber_yield

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
    pattern = r"FPI_(\d+(?:\.\d+)?)(?:cm)?"
    print(pattern)
