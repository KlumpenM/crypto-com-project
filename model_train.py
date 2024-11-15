# Import the logistic regression model from sklearn
from sklearn.linear_model import LogisticRegression
# Import the fetch_openml function from sklearn.datasets
from sklearn.datasets import fetch_openml
# Import the train_test_split function from sklearn.model_selection
from sklearn.model_selection import train_test_split
# Import the StandardScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load the MNIST dataset
print('Loading the MNIST dataset...')
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist.data, mnist.target.astype(int)

# Step 2: process the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train the logistic regression model
print('Training the logistic regression model...')
model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Step 5: Save the trained model
print('Saving the trained model...')
joblib.dump(model, 'logistic_model_mnist.joblib')
print('Model trained and saved')


# Load the model
print('Loading the trained model...')
model = joblib.load('logistic_model_mnist.joblib')

# Predict on test data
predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f'Accuracy: {accuracy * 100:.2f}')



import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Use one test sample for prediction
sample_date = X_test[0] # Pick one test sample
true_label = y_test.iloc[0] # Get the true label of the sample

# Extract model parameters
weights = model.coef_
bias = model.intercept_

# Perform forward pass
logits = np.dot(sample_date, weights.T) + bias
probabilities = sigmoid(logits)
predicted_class = np.argmax(probabilities)

# Printing statements for output
print(f'True label: {true_label}')
print(f'Predicted Class: {predicted_class}')
print(f'Probabilities: {probabilities}')