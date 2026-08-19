# Task 1 - Primary Random Forest Training
from sklearn.metrics import accuracy_score

# 1. Initialize and fit Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# 2. Predict on Training and Validation sets
y_train_pred = rf_model.predict(X_train_scaled)
y_val_pred = rf_model.predict(X_test_scaled)

# 3. Calculate Accuracies
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_test, y_val_pred)

print(f"Training Accuracy   : {train_acc * 100:.2f}%")
print(f"Validation Accuracy : {val_acc * 100:.2f}%")