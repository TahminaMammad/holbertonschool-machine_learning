#!/usr/bin/env python3
import pandas as pd
def from_numpy(array):
    num_cols = array.shape[1]
    columns = [chr(i) for i in range(65, 65 + num_cols)]
    return pd.DataFrame(array, columns=columns)
