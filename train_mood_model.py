"""
AI Music Mood Classification - Model Training (Calm + Happy Merged)
----------------------------------------------------------------------
Trains a Random Forest classifier to predict song mood.
Calm and Happy are combined into a single class: Calm_Happy.
Final classes: Calm_Happy, Sad, Energetic

Outputs:
- Training accuracy
- Validation accuracy
- Confusion matrix
- Classification report
- Feature importance
- Saved trained model
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

import joblib


# ============================================================
# 1. SET FOLDER PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "music_mood_dataset.csv"
)
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CHECK DATASET
# ============================================================

print("=" * 60)
print("AI MUSIC MOOD CLASSIFICATION")
print("=" * 60)

print("\nDataset path:")
print(DATA_PATH)

if not os.path.exists(DATA_PATH):
    print("\nERROR: Dataset not found!")
    print("\nPlease make sure the CSV file is located at:")
    print(DATA_PATH)
    exit()


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nOriginal mood class distribution:")
print(df["Mood"].value_counts())


# ============================================================
# 4. MERGE CALM + HAPPY INTO ONE CLASS
# ============================================================

MOOD_MAP = {
    "Calm": "Calm_Happy",
    "Happy": "Calm_Happy",
    "Sad": "Sad",
    "Energetic": "Energetic",
}

df["Mood"] = df["Mood"].map(MOOD_MAP)

print("\nMood class distribution after merging Calm + Happy:")
print(df["Mood"].value_counts())


# ============================================================
# 5. SELECT FEATURES
# ============================================================

FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
]

TARGET = "Mood"


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: The following columns are missing:")

    for column in missing_columns:
        print("-", column)

    print("\nColumns in the dataset:")
    print(df.columns.tolist())

    exit()


# ============================================================
# 7. REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=FEATURES + [TARGET]
)

print("\nDataset after removing missing values:")
print(df.shape)


# ============================================================
# 8. CREATE FEATURES AND TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]

print("\nFeatures:")
print(FEATURES)

print("\nTarget:")
print(TARGET)

print("\nMood classes:")
print(sorted(y.unique()))


# ============================================================
# 9. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set size:")
print(X_train.shape[0])

print("Validation set size:")
print(X_val.shape[0])


# ============================================================
# 10. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=18,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 11. TRAINING ACCURACY
# ============================================================

train_preds = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_preds
)

print("\nTraining Accuracy:")
print(f"{train_accuracy:.4f}")


# ============================================================
# 12. VALIDATION ACCURACY
# ============================================================

val_preds = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_preds
)

print("\nValidation Accuracy:")
print(f"{val_accuracy:.4f}")


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_val,
        val_preds
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_val,
    val_preds,
    labels=labels
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print(cm_df)


# ============================================================
# 15. SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True
)

plt.title(
    "Confusion Matrix - Music Mood Classifier"
)

plt.tight_layout()

confusion_matrix_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix_merged.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=150
)

plt.close()

print("\nConfusion matrix saved:")
print(confusion_matrix_path)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

importances = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importances)


# ============================================================
# 17. SAVE FEATURE IMPORTANCE GRAPH
# ============================================================

fig2, ax2 = plt.subplots(
    figsize=(8, 5)
)

sns.barplot(
    x=importances.values,
    y=importances.index,
    ax=ax2,
    color="steelblue"
)

ax2.set_xlabel("Importance")
ax2.set_ylabel("Feature")

ax2.set_title(
    "Feature Importance - Random Forest"
)

plt.tight_layout()

feature_importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance_merged.png"
)

plt.savefig(
    feature_importance_path,
    dpi=150
)

plt.close()

print("\nFeature importance saved:")
print(feature_importance_path)


# ============================================================
# 18. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "mood_model_merged.pkl"
)

joblib.dump(
    model,
    model_path,
    compress=3
)

print("\nTrained model saved:")
print(model_path)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"Training Accuracy:   {train_accuracy:.4f}"
)

print(
    f"Validation Accuracy: {val_accuracy:.4f}"
)

print("\nOutput files:")
print("- confusion_matrix_merged.png")
print("- feature_importance_merged.png")
print("- mood_model_merged.pkl")

print("\nTraining completed successfully!")