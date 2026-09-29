import curie as ci
import numpy as np
import pandas as pd
import subprocess

Det2_isotopes = ['198AU', '198HG', '196AU', '196PT', '116IN', '116SN', '113IN', '115IN', '111CD', '114IN', '114CD', '57NI', '57CO', '57FE', '58CO', '58FE', '60CO', '60NI', '87Y', '87SR', '88Y', '88SR', '89ZR', '89Y']

# 15.00 group
Ba133_IDM_15 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_133Ba_15cm.Spe")
Ba133_IDM_15.isotopes = ['133BA']
Cs137_IDM_15 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_137Cs_15cm.Spe")
Cs137_IDM_15.isotopes = ['137CS']
Eu152_IDM_15 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_152Eu_15cm.Spe")
Eu152_IDM_15.isotopes = ['152EU']
Ba133_IDM_15.plot()
Cs137_IDM_15.plot()
Eu152_IDM_15.plot()
cb_15 = ci.Calibration()
cb_15.calibrate([Ba133_IDM_15, Cs137_IDM_15, Eu152_IDM_15], sources=[{'isotope':'133BA', 'A0':38590.0, 'ref_date':'03/01/2019 12:00:00'}, {'isotope':'137CS', 'A0':36752.84, 'ref_date':'02/08/2019 12:00:00'}, {'isotope':'152EU', 'A0':3.822E4, 'ref_date':'03/01/2019 12:00:00'}])

# 18.00 group
Ba133_IDM_18 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_133Ba_18cm.Spe")
Ba133_IDM_18.isotopes = ['133BA']
Cs137_IDM_18 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_137Cs_18cm.Spe")
Cs137_IDM_18.isotopes = ['137CS']
Eu152_IDM_18 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_152Eu_18cm.Spe")
Eu152_IDM_18.isotopes = ['152EU']
Ba133_IDM_18.plot()
Cs137_IDM_18.plot()
Eu152_IDM_18.plot()
cb_18 = ci.Calibration()
cb_18.calibrate([Ba133_IDM_18, Cs137_IDM_18, Eu152_IDM_18], sources=[{'isotope':'133BA', 'A0':38590.0, 'ref_date':'03/01/2019 12:00:00'}, {'isotope':'137CS', 'A0':36752.84, 'ref_date':'02/08/2019 12:00:00'}, {'isotope':'152EU', 'A0':3.822E4, 'ref_date':'03/01/2019 12:00:00'}])

# 30.00 group
Ba133_IDM_30 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260923_Det2_133Ba_30cm.Spe")
Ba133_IDM_30.isotopes = ['133BA']
Cs137_IDM_30 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_137Cs_30cm.Spe")
Cs137_IDM_30.isotopes = ['137CS']
Eu152_IDM_30 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_152Eu_30cm.Spe")
Eu152_IDM_30.isotopes = ['152EU']
Ba133_IDM_30.plot()
Cs137_IDM_30.plot()
Eu152_IDM_30.plot()
cb_30 = ci.Calibration()
cb_30.calibrate([Ba133_IDM_30, Cs137_IDM_30, Eu152_IDM_30], sources=[{'isotope':'133BA', 'A0':38590.0, 'ref_date':'03/01/2019 12:00:00'}, {'isotope':'137CS', 'A0':36752.84, 'ref_date':'02/08/2019 12:00:00'}, {'isotope':'152EU', 'A0':3.822E4, 'ref_date':'03/01/2019 12:00:00'}])

# 50.00 group
Ba133_IDM_50 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260923_Det2_133Ba_50cm.Spe")
Ba133_IDM_50.isotopes = ['133BA']
Cs137_IDM_50 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_137Cs_50cm.Spe")
Cs137_IDM_50.isotopes = ['137CS']
Eu152_IDM_50 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260924_Det2_152Eu_50cm.Spe")
Eu152_IDM_50.isotopes = ['152EU']
Ba133_IDM_50.plot()
Cs137_IDM_50.plot()
Eu152_IDM_50.plot()
cb_50 = ci.Calibration()
cb_50.calibrate([Ba133_IDM_50, Cs137_IDM_50, Eu152_IDM_50], sources=[{'isotope':'133BA', 'A0':38590.0, 'ref_date':'03/01/2019 12:00:00'}, {'isotope':'137CS', 'A0':36752.84, 'ref_date':'02/08/2019 12:00:00'}, {'isotope':'152EU', 'A0':3.822E4, 'ref_date':'03/01/2019 12:00:00'}])

# 80.00 group
Ba133_IDM_80 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260922_Det2_133Ba_80cm.Spe")
Ba133_IDM_80.isotopes = ['133BA']
Cs137_IDM_80 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260916_Det2_137Cs_80cm.Spe")
Cs137_IDM_80.isotopes = ['137CS']
Eu152_IDM_80 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260917_Det2_152Eu_80cm.Spe")
Eu152_IDM_80.isotopes = ['152EU']
Ba133_IDM_80.plot()
Cs137_IDM_80.plot()
Eu152_IDM_80.plot()
cb_80 = ci.Calibration()
cb_80.calibrate([Ba133_IDM_80, Cs137_IDM_80, Eu152_IDM_80], sources=[{'isotope':'133BA', 'A0':38590.0, 'ref_date':'03/01/2019 12:00:00'}, {'isotope':'137CS', 'A0':36752.84, 'ref_date':'02/08/2019 12:00:00'}, {'isotope':'152EU', 'A0':3.822E4, 'ref_date':'03/01/2019 12:00:00'}])

# 138.00 group
Ba133_IDM_138 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260921_Det2_133Ba_138cm.Spe")
Ba133_IDM_138.isotopes = ['133BA']
Cs137_IDM_138 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260916_Det2_137Cs_138cm.Spe")
Cs137_IDM_138.isotopes = ['137CS']
Eu152_IDM_138 = ci.Spectrum("/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Calibrations BG/Det2/20260917_Det2_152Eu_138cm.Spe")
Eu152_IDM_138.isotopes = ['152EU']
Ba133_IDM_138.plot()
Cs137_IDM_138.plot()
Eu152_IDM_138.plot()
cb_138 = ci.Calibration()
cb_138.calibrate([Ba133_IDM_138, Cs137_IDM_138, Eu152_IDM_138], sources=[{'isotope':'133BA', 'A0':38590.0, 'ref_date':'03/01/2019 12:00:00'}, {'isotope':'137CS', 'A0':36752.84, 'ref_date':'02/08/2019 12:00:00'}, {'isotope':'152EU', 'A0':3.822E4, 'ref_date':'03/01/2019 12:00:00'}])

####################################

cb_15.saveas('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Det2_calib/Det2_cb_15.json')
cb_18.saveas('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Det2_calib/Det2_cb_18.json')
cb_30.saveas('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Det2_calib/Det2_cb_30.json')
cb_50.saveas('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Det2_calib/Det2_cb_50.json')
cb_80.saveas('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Det2_calib/Det2_cb_80.json')
cb_138.saveas('/home/james_andrew/Python_Projects/Pb-212/August_BeamTime/data/Det2_calib/Det2_cb_138.json')

