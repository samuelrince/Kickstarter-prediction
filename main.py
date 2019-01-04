import pandas as pd
from sklearn.linear_model import LogisticRegression


#############
# Load data #
#############
# ===== Load csv file =====
length_training_set = 250000
data = pd.read_csv(filepath_or_buffer='ks-projects-201801.csv', delimiter=',', header=0)
data = data.dropna()    # Drop all rows that has NaN values
train_set = data.head(length_training_set)
test_set = data.tail(data.__len__() - length_training_set)

# ===== Data generation =====
# Feature selection
features = ['category', 'main_category', 'currency', 'deadline', 'launched', 'country',
            'usd pledged', 'usd_goal_real']
X_train = pd.DataFrame(data=train_set[features])
X_test = pd.DataFrame(data=test_set[features])
# Labels
y_train = pd.DataFrame(data=train_set['state'])
y_test = pd.DataFrame(data=test_set['state'])


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
for ctr in data['country']:
    if ctr not in country:
        country[ctr] = index
        index += 1
# Labels
labels = dict()
for label in data['state']:
    if label not in country:
        if label == 'successful':
            labels[label] = 1
        else:
            labels[label] = 0

# ===== Update data values =====
# Main category
X_train['main_category'].replace(main_category, inplace=True)
X_test['main_category'].replace(main_category, inplace=True)
# Sub category
X_train['category'].replace(sub_category, inplace=True)
X_test['category'].replace(sub_category, inplace=True)
# Currency
X_train['currency'].replace(currency, inplace=True)
X_test['currency'].replace(currency, inplace=True)
# Country
X_train['country'].replace(country, inplace=True)
X_test['country'].replace(country, inplace=True)

# ===== Duration calculation =====
# Convert data to datetime format
X_train['deadline'] = pd.to_datetime(X_train['deadline'], format='%Y-%m-%d')
X_test['deadline'] = pd.to_datetime(X_test['deadline'], format='%Y-%m-%d')
X_train['launched'] = pd.to_datetime(X_train['launched'], format='%Y-%m-%d %H:%M:%S')
X_test['launched'] = pd.to_datetime(X_test['launched'], format='%Y-%m-%d %H:%M:%S')
# Adding duration time to the data in days
X_train['duration'] = (X_train['deadline']-X_train['launched']).astype('timedelta64[D]')
X_test['duration'] = (X_test['deadline']-X_test['launched']).astype('timedelta64[D]')
# Delete 'launched' and 'deadline' columns
X_train.drop(columns=['deadline', 'launched'], inplace=True)
X_test.drop(columns=['deadline', 'launched'], inplace=True)

# ===== Labels =====
y_train['state'].replace(labels, inplace=True)
y_test['state'].replace(labels, inplace=True)
