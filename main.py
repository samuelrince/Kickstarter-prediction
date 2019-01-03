import numpy as np
import pandas as pd

#############
# Load data #
#############
length_training_set = 1000
data = pd.read_csv(filepath_or_buffer='ks-projects-201801.csv', delimiter=',', header=0)
train_set = data.head(length_training_set)
test_set = data.tail(data.__len__() - length_training_set)

# ===== Data generation =====
# Feature selection
features = ['name', 'category', 'main_category', 'currency', 'deadline', 'launched', 'backers', 'country',
            'usd pledged', 'usd_goal_real']
X_train = train_set[features]
X_test = test_set[features]
# Labels
y_train = train_set['state']
y_test = test_set['state']