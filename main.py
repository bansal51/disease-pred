import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


training_data_path = "dataset/Training.csv"
# Drop the ending column --> empty column
training_data = pd.read_csv(training_data_path).dropna(axis=1)

# Check to see whether the dataset is balanced
disease_count = training_data["prognosis"].value_counts()
# We are comparing the various diseases and making sure each of them have equal amount of prognosis values
temp = pd.DataFrame({"Disease": disease_count.index, "Counts": disease_count.values})
plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp)
plt.xticks(rotation=90)
# Uncomment this to see evidence of balanced data
# plt.show()

# After this, we can see that all of the columns have an equal amount of prognosis counts, so the data is balanced.
# Our target column (prognosis) are Strings, so we have to convert the prognosis column into a numerical value using an encoder
# Label Encoder assigns a numerical value to the various prognoses --> if there are n labals, the labels will be assigned from 0 to n - 1
encoder = LabelEncoder()
training_data["prognosis"] = encoder.fit_transform(training_data["prognosis"])

# The X group should be all the columns used to determine the prognosis (independent variables)
# The y group should be the columns which is the prognosis (dependent variable)
# Test data is only 42 entries, we want a 80/20 split between the training and test data --> we will split the training data
X = training_data.iloc[:, :-1]
y = training_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now that the data is split, we will use 3 ML models (Naive Bayes, Random Forest, and Support Vector Classifier) to get 3 predictions
# Using these 3 predictions, we will use k-fold cross validation to evalute the accuracy of the 3 models
# We need to define a scoring metric for the cross validation
def cross_validation_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Init the 3 models
models = {
    # The SVM algorithm tries to find the optimal hyperplane that can separate the datapoints in different classes
    "SVC": SVC(),
    # Naive Bayes methods use Bayes formula with the assumption that all features have conditional independence. We use a Gaussian (normal) distribution
    "Naive Bayes": GaussianNB(),
    # Random Forest is a group-like classifier which combines the results of many decision trees to make a better and more accurate model
    "Random Forest": RandomForestClassifier(random_state=18)
}

# Producing a cross-validation score for each of the models
# From the output, we see that the models are performing well, and the mean scores after k-fold are high
# We can combine the mode of all the predictions by the model so that the results will be more accurate on unknown data
for model in models:
    name = models[model]
    scores = cross_val_score(name, X, y, cv=10, n_jobs=-1, scoring=cross_validation_scoring)

    # Uncomment to see cross validation between 3 models
    # print("=="*30)
    # print(model)
    # print(f"Scores: {scores}")
    # print(f"Mean Score: {np.mean(scores)}")

# Now, we train and test the SVM classifier
# First we fit the model with the training data, and then test and compare using the test data
svm_model = SVC()
svm_model.fit(X_train, y_train)
prediction = svm_model.predict(X_test)

# We now compare the accuracy on the training and test data
# print(f"Accuracy on train data by SVM Classifier: {accuracy_score(y_train, svm_model.predict(X_train)) *  100}")
# print(f"Accuracy on test data by SVM Classifier: {accuracy_score(y_test, prediction) * 100}")

cf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
# plt.show()

# Now, we train and test the Gaussian NB classifier
# First we fit the model with the training data, and then test and compare using the test data
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
prediction = gnb_model.predict(X_test)

# We now compare the accuracy on the training and test data
# print(f"Accuracy on train data by GNB Classifier: {accuracy_score(y_train, gnb_model.predict(X_train)) *  100}")
# print(f"Accuracy on test data by GNB Classifier: {accuracy_score(y_test, prediction) * 100}")

cf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for GNB Classifier on Test Data")
# plt.show()

# Now, we train and test the Random Forest classifier
# First we fit the model with the training data, and then test and compare using the test data
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
prediction = rf_model.predict(X_test)

# We now compare the accuracy on the training and test data
# print(f"Accuracy on train data by RF Classifier: {accuracy_score(y_train, rf_model.predict(X_train)) *  100}")
# print(f"Accuracy on test data by RF Classifier: {accuracy_score(y_test, prediction) * 100}")

cf_matrix = confusion_matrix(y_test, prediction)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for RF Classifier on Test Data")
# plt.show()

# From the above confusion matrices, we see that the models are performing very well on the unseen data.
# Now we will use the full training data to train the models and then test on the actual test data
updated_svm_model = SVC()
updated_gnb_model = GaussianNB()
updated_rf_model = RandomForestClassifier(random_state=18)

# We are now fitting the updated models with the full training data
updated_svm_model.fit(X, y)
updated_gnb_model.fit(X, y)
updated_rf_model.fit(X, y)

# Read in the test data
test_data_path = "dataset/Testing.csv"
test_data = pd.read_csv(test_data_path).dropna(axis=1)
test_X = test_data.iloc[:, :-1]
test_y = encoder.transform(test_data.iloc[:, -1])

# Making prediction by taking mode of predictions made by all the classifiers
svm_prediction = updated_svm_model.predict(test_X)
gnb_prediction = updated_gnb_model.predict(test_X)
rf_prediction = updated_rf_model.predict(test_X)

final_predictions = [mode([i, j, k])[0] for i,j,k in zip(svm_prediction, gnb_prediction, rf_prediction)]
print(final_predictions)
print(f"Accuracy on test dataset by the combined model: {accuracy_score(test_y, final_predictions) * 100}")
cf_matrix = confusion_matrix(test_y, final_predictions)

# The confusion matrix classifies the data points accurately
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Combined Model on Test Data")
# plt.show()