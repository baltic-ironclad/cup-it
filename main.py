import os
import pprint

import pandas as pd
import numpy as np
import matplotlib as mpl

df = pd.read_csv(os.path.join('data', 'train_data.csv'), encoding='utf-8')
df['type'].replace(['Обслуживание физ. и юр. лиц'], ['Обслуживание лиц'], inplace=True)
df.sort_values(by='type', ascending=True, inplace=True)
