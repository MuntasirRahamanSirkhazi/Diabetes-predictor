import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load dataset
data = pd.read_csv("data/diabetes.csv")

# 2. Features (X) and target (y)
X = data.drop("Outcome", axis=1)   # all columns except Outcome
y = data["Outcome"]               # 0 = No Diabetes, 1 = Diabetes

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Check accuracy (optional)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# 6. Save trained model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as diabetes_model.pkl")
