import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#############
# Load data #
#############
# ===== Load csv file =====
data = pd.read_csv(filepath_or_buffer='ks-projects-201801.csv', delimiter=',', header=0)
data = data.dropna()    # Drop all rows that has NaN values

# ===== Data generation =====
# Feature selection
features = ['category', 'main_category', 'currency', 'deadline', 'launched', 'country', 'usd_goal_real']
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
#    if label not in country:
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
X.drop(labels=['deadline', 'launched'],axis=1, inplace=True)

# ===== Labels =====
y['state'].replace(labels, inplace=True)


####################
# Train-Test split #
####################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#######################
# Logistic Regression #
#######################

ss = StandardScaler()
lr = LogisticRegression()
lr_pipe = Pipeline([('sscale', ss), ('logreg', lr)])
lr_pipe.fit(X_train, y_train.values.ravel())
print('Logistic Regression acc:', lr_pipe.score(X_test, y_test))

# LR Accuracy: 0.644578


############################
# Decision Tree Classifier #
############################

dtc = DecisionTreeClassifier(max_depth=None, max_features='auto')
dtc.fit(X_train, y_train)
print('Decision Tree Classifier acc:', dtc.score(X_test, y_test))

# DTC Accuracy: 0.650787


#################
# Random Forest #
#################

rf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features='auto')
rf.fit(X_train, y_train.values.ravel())
y_predict = rf.predict(X_test)
print('Random Forest acc:', accuracy_score(y_test, y_predict))

# RF Accuracy: 0.665790 (n_estimators=50)
# RF Accuracy: 0.667383 (n_estimators=100)
# RF Accuracy: 0.667569 (n_estimators=200)
# RF Accuracy: 0.668175 (n_estimators=1000)


#######
# kNN #
#######

neigh = KNeighborsClassifier(n_neighbors=50)
neigh.fit(X_train, y_train.values.ravel())
y_predict = neigh.predict(X_test)
print('kNN acc:', accuracy_score(y_test, y_predict))

# kNN Accuracy: 0.634514 (n_neighbors=3)
# kNN Accuracy: 0.664384 (n_neighbors=10)
# kNN Accuracy: 0.664893 (n_neighbors=20)
# kNN Accuracy: 0.669226 (n_neighbors=50)
# kNN Accuracy: 0.660956 (n_neighbors=200)


############
# AdaBoost #
############

def AdaBoost(D, T,X_train,y_train,X_test,y_test):
    X_train = X_train.values
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    for i in range(len(y_test)): #transform results to have positive and negative values
        if(y_test[i] == 0):             #instead of 0 and 1
            y_test[i] = -1
    for i in range(len(y_train)):  #transform results to have positive and negative values
        if(y_train[i] == 0):
            y_train[i] = -1
    w = np.ones(X_train.shape[0]) / X_train.shape[0]
    training_scores = np.zeros(X_train.shape[0])
    test_scores = np.zeros(X_test.shape[0])
    training_errors = []
    test_errors = []

    for t in range(T):
        clf = RandomForestClassifier(n_estimators=10, max_depth=D, max_features='auto')
        clf.fit(X_train, y_train, sample_weight=w)
        y_pred = clf.predict(X_train)
        
        indicator = np.not_equal(y_pred, y_train)
        gamma = w[indicator].sum() / w.sum()
        alpha = np.log((1-gamma) / gamma)
        w *= np.exp(alpha * indicator) 

        training_scores += alpha * y_pred
        training_error = 1. * len(training_scores[training_scores * y_train < 0]) / len(X_train)
        y_test_pred = clf.predict(X_test) # Examine the test set
        test_scores += alpha * y_test_pred
        test_error = 1. * len(test_scores[test_scores * y_test < 0]) / len(X_test)

        plt.clf()
        training_errors.append(training_error)
        test_errors.append(test_error)
        return test_errors, training_errors, test_error

    
### Test with one value of D
    
D = 13 # Depth of trees
T = 10 # Number of iterations of AdaBoost
test_errors, training_errors, result_error = AdaBoost(D,T,X_train,y_train,X_test,y_test)
print("Accuracy with AdaBoost :", 1 - result_error)

plt.plot(training_errors, label="training error")
plt.plot(test_errors, label="test error")
plt.legend()
plt.show()

##### Test with several values of D
#Ds = [2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15,16,17,18,19,20]
#final_test_errors = []
#for D in Ds:
#    print(D)
#    final_test_errors.append(AdaBoost(D, 10,X_train,y_train,X_test,y_test)[2])
#
#
#plt.plot(Ds, final_test_errors)
#plt.title("Test error vs. tree depth")
#plt.ylabel("Test error")
#plt.xlabel("Tree depth")
#plt.show()

#####################
# Overfitting tests #
#####################

train_number = range(10000,300000,10000)
n = len(train_number)
train_errors = np.zeros(n)
test_errors= np.zeros(n)
for t in range(n) :
    print(t)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    X_train = X_train.head(train_number[t])
    y_train = y_train.head(train_number[t])
    dtc = DecisionTreeClassifier(max_depth=None, max_features='auto')
    dtc.fit(X_train, y_train)
    test_errors[t] = 1 - dtc.score(X_test, y_test)
    train_errors[t] = 1 - dtc.score(X_train, y_train)
    

plt.plot(train_number,train_errors,label = "train" )
plt.plot(train_number,test_errors,label = "test")
plt.title("error vs. number of train data with Decision Tree Classifier")
plt.xlabel("Number of train data")
plt.ylabel("Error")
plt.legend()
plt.show()
