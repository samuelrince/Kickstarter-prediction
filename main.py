import pandas as pd

#############
# Load data #
#############
# ===== Load csv file =====
length_training_set = 1000
data = pd.read_csv(filepath_or_buffer='ks-projects-201801.csv', delimiter=',', header=0)
train_set = data.head(length_training_set)
test_set = data.tail(data.__len__() - length_training_set)

# ===== Data generation =====
# Feature selection
features = ['category', 'main_category', 'currency', 'deadline', 'launched', 'country',
            'usd pledged', 'usd_goal_real']
X_train = train_set[features]
X_test = test_set[features]
# Labels
y_train = train_set['state']
y_test = test_set['state']


#######################
# Data pre-processing #
#######################
# ===== Adapt data values =====
# Main category
main_category, index = dict(), 0
for cat in data['main_category']:
    if cat not in main_category:
        main_category[cat] = index
        index += 1
# Sub category
sub_category, index = dict(), 0
for cat in data['category']:
    if cat not in sub_category:
        sub_category[cat] = index
        index += 1
# Currency
currency, index = dict(), 0
for cur in data['currency']:
    if cur not in currency:
        currency[cur] = index
        index += 1
# Country
country, index = dict(), 0
for cur in data['country']:
    if cur not in country:
        country[cur] = index
        index += 1

