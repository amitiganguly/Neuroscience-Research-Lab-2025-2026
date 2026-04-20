#importing libraries
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA


#load the BDF file
# change the hash on line 10 with the correct participant number
file_name = "P#.bdf"
raw = mne.io.read_raw_bdf(file_name, preload=True)


#channel mappings
channel_mappings = {
   'small': {'EEG 4': 'P4','EEG 6': 'Fz','EEG 7': 'P3'},
   'medium': {'EEG 1': 'P3','EEG 2': 'Fz','EEG 3': 'P4'},
   'large': {'EEG 1': 'P3','EEG 5': 'P4','EEG 8': 'Fz'}
}


headset_size = input("Enter headset size (small, medium, large): ").strip().lower()
if headset_size not in channel_mappings:
   raise ValueError("Invalid headset size!")


raw.rename_channels(channel_mappings[headset_size])
raw.pick(['Fz', 'P3', 'P4'])


#montage + filtering
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)


raw.filter(l_freq=1.0, h_freq=None)
raw.notch_filter(freqs=60)


#ICA
ica = ICA(n_components=0.95, method="fastica", random_state=97, max_iter="auto")
ica.fit(raw)


eog_indices, _ = ica.find_bads_eog(raw, ch_name='Fz')
ica.exclude.extend(eog_indices)


raw_clean = ica.apply(raw.copy())


#segment times
b1_start = float(input("Baseline 1 start: "))
b1_end = float(input("Baseline 1 end: "))
e1_start = float(input("Experimental 1 start: "))
e1_end = float(input("Experimental 1 end: "))


b2_start = float(input("Baseline 2 start: "))
b2_end = float(input("Baseline 2 end: "))
e2_start = float(input("Experimental 2 start: "))
e2_end = float(input("Experimental 2 end: "))


#crop segments
baseline1 = raw_clean.copy().crop(tmin=b1_start, tmax=b1_end)
experimental1 = raw_clean.copy().crop(tmin=e1_start, tmax=e1_end)
baseline2 = raw_clean.copy().crop(tmin=b2_start, tmax=b2_end)
experimental2 = raw_clean.copy().crop(tmin=e2_start, tmax=e2_end)


#determine order
beat_first = input("Which beat first? (theta/gamma): ").strip().lower()
if beat_first not in ['theta', 'gamma']:
   raise ValueError("Invalid input!")


if beat_first == "theta":
   conditions = {
       "7 Hz": {"baseline": baseline1, "experimental": experimental1},
       "40 Hz": {"baseline": baseline2, "experimental": experimental2}
   }
else:
   conditions = {
       "40 Hz": {"baseline": baseline1, "experimental": experimental1},
       #"7 Hz": {"baseline": baseline2, "experimental": experimental2}
   }


#frequency bands
bands = {
   "theta": (4, 8),
   "alpha": (8, 12),
   "beta": (13, 30),
   "gamma": (30, 50)
}


#function to compute RMS
def compute_band_summary(segment, l_freq, h_freq):
   temp = segment.copy()
   temp.filter(
       l_freq=l_freq,
       h_freq=h_freq,
       method='iir',
       iir_params={'order': 6, 'ftype': 'butter'},
       phase='zero'
   )
   data = temp.get_data() * 1e6  #convert to µV
   df = pd.DataFrame(data.T, columns=temp.ch_names)
   return df.describe()


#MAIN OUTPUT
print("\nParticipant:", file_name)


for cond_name, phases in conditions.items():
   print(f"\n==============================")
   print(f"{cond_name} CONDITION")
   print(f"==============================")


   for phase_name, segment in phases.items():
       print(f"\n--- {phase_name.upper()} ---")


       for band_name, (low, high) in bands.items():
           print(f"\n{band_name.upper()} BAND ({low}-{high} Hz):")
           summary = compute_band_summary(segment, low, high)
           print(summary)

