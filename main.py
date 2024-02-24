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
print(training_data)

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
