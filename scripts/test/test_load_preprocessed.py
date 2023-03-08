

import numpy as np

d = np.load('./data/fma_preprocessed/coarse/train_0.npy', allow_pickle=True)

for key, val in d.item().items():
    print(key, val.shape)