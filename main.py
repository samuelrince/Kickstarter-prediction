import pandas as pd
from sklearn.linear_model import LogisticRegression
from logistic_regression_classifier import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#############
# Load data #
#############
# ===== Load csv file =====
data = pd.read_csv(filepath_or_buffer='ks-projects-201801.csv', delimiter=',', header=0)
data = data.dropna()    # Drop all rows that has NaN values
# data = data.head(15000)
# print(data.shape)

# ===== Data generation =====
# Feature selection
features = ['category', 'main_category', 'currency', 'deadline', 'launched', 'country',
            'usd pledged', 'usd_goal_real']
X = pd.DataFrame(data=data[features])
# Labels
y = pd.DataFrame(data=data['state'])


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
X['main_category'].replace(main_category, inplace=True)
# Sub category
X['category'].replace(sub_category, inplace=True)
# Currency
X['currency'].replace(currency, inplace=True)
# Country
X['country'].replace(country, inplace=True)

# ===== Duration calculation =====
# Convert data to datetime format
X['deadline'] = pd.to_datetime(X['deadline'], format='%Y-%m-%d')
X['launched'] = pd.to_datetime(X['launched'], format='%Y-%m-%d %H:%M:%S')
# Adding duration time to the data in days
X['duration'] = (X['deadline']-X['launched']).astype('timedelta64[D]')
# Delete 'launched' and 'deadline' columns
X.drop(columns=['deadline', 'launched'], inplace=True)

# ===== Labels =====
y['state'].replace(labels, inplace=True)


####################
# Train-Test split #
####################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#######################
# Logistic regression #
#######################
# W, projected_centroid, X_lda = logistic_regression_classifier(X_train.values, y_train.values)
# predictedLabels_LDA = predict(X_test.values, projected_centroid, W)
# total_correct = 0
# total_incorrect = 0
# for i in range(len(predictedLabels_LDA)):
#     if predictedLabels_LDA[i] == y_test.values[i][0]:
#         total_correct += 1
#     else:
#         total_incorrect += 1
# print('Accuracy logistic regression :', total_correct / (total_correct + total_incorrect))


#######################
# Logistic regression #
#######################
ss = StandardScaler()
lr = LogisticRegression()
lr_pipe = Pipeline([('sscale', ss), ('logreg', lr)])
lr_pipe.fit(X_train, y_train)
print('lr acc:', lr_pipe.score(X_test, y_test))


#################
# Random Forest #
#################
rf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features='auto')
rf.fit(X_train, y_train.values.ravel())
y_predict = rf.predict(X_test)
print('rf acc:', accuracy_score(y_test, y_predict))


#################
# Decision Tree #
#################
dtc = DecisionTreeClassifier(max_depth=None, max_features='auto')
dtc.fit(X_train, y_train)
print('dtc acc:', dtc.score(X_test, y_test))