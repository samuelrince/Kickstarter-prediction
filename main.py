import numpy as np
import pandas as pd

#############
# Load data #
#############
len_training_set = 1000
data = pd.read_csv(filepath_or_buffer='ks-projects-201801.csv', delimiter=',', header=0)
train_set = data.head(len_training_set)
test_set = data.tail(data.__len__() - len_training_set)
