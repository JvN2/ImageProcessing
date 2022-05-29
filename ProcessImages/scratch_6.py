import numpy as np
import pandas as pd
import fnmatch

import ProcessImages.ImageIO as im3
import matplotlib.pyplot as plt
from ProcessImages.TraceAnalysis import Traces
from tqdm import tqdm

filename = r'C:\Users\jvann\surfdrive\werk\Data\CoSMoS\tmp.xlsx'

data = Traces(filename)
selection = fnmatch.filter(data.traces.columns, '*: I * (a.u.)')

for trace in tqdm(data.traces[selection], postfix='Fit HMM'):
    print(trace)
