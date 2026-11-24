import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import Neutron_Flux_Analysis.Serber_Yield_Calculations as nyld

"""
I want to develop a program that will track neutrons through the PLA target holder, mark collisions and tally energies deposited as well as mark 12C(n,2n)11C reactions that
produce B+ decay, this deposits ~1MeV through positron anhilation. I want to start by tracking the first couple scatters if the collision is not with an electron since the
neutron will retain most of its energy

Two spaces will be developed, the center of the PLA {where the foils were} and the surrounding PLA structure that I will model, I will assume neutrons that scatter into the
center will be lost. Initial neutron energies and angles will be determined by the serber breakup code.

I will need to determine the relevant cross sections and draw a linear interpolation between the points to have a smooth cross section function
"""