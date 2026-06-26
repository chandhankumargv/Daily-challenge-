import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("J26parkinsons.csv")

# Remove name column
X = data.drop(columns=["name", "status"], axis=1)
Y = data["status"]

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=2,
    stratify=Y
)

# Standardize Data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = SVC(kernel='linear')

model.fit(X_train, Y_train)

# Accuracy
train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_prediction)

test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_prediction)

print("Training Accuracy :", train_accuracy)
print("Testing Accuracy  :", test_accuracy)

# Example Prediction
input_data = (
    119.99200,157.30200,74.99700,0.00784,0.00007,
    0.00370,0.00554,0.01109,0.04374,0.42600,
    0.02182,0.03130,0.02971,0.06545,0.02211,
    21.03300,0.414783,0.815285,-4.813031,
    0.266482,2.301442,0.284654
)

input_array = np.asarray(input_data).reshape(1, -1)

input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)

if prediction[0] == 0:
    print("The person does NOT have Parkinson's Disease.")
else:
    print("The person HAS Parkinson's Disease.")
