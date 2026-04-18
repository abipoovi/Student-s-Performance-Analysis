import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Step 1: Create dataset
data = pd.DataFrame({
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 90]
})

# Step 2: Features and Target
X = data[["Study_Hours", "Attendance"]]
y = data["Marks"]

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict test data
y_pred = model.predict(X_test)

# Step 6: Evaluate model
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))

# Step 7: Predict new student performance (FIXED - no warning)
new_data = pd.DataFrame({
    "Study_Hours": [5],
    "Attendance": [80]
})

predicted_marks = model.predict(new_data)
print("Predicted Marks:", predicted_marks[0])

# Step 8: Visualization (Study Hours vs Marks)
plt.scatter(data["Study_Hours"], data["Marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()
