import numpy as np
import pandas as pd
import pickle
import gzip
import os

def fix_data(folder):
    os.chdir(folder)
    for f in os.listdir():
        if '.data' in f:
            fp=gzip.open(f, 'rb')
            data=pickle.load(fp)
            fp.close()
            
            new_f = f.replace('.data', '.npz')
            np.savez_compressed(new_f,
                            masks=np.array(data['masks']),
                            boxes=np.array(data['boxes']),
                            scores=np.array(data['scores']),
                            sample_counts=np.array(data['sample_counts']))
