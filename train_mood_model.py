"""
AI Music Mood Classification - Model Training
------------------------------------------------
Trains a Random Forest classifier to predict song mood:
Calm, Happy, Sad, Energetic

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

# Folder tempat train_mood_model.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV berada dalam folder yang sama dengan Python file
DATA_PATH = os.path.join(
    BASE_DIR,
    "music_mood_dataset.csv"
)

# Folder output
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

# Create outputs folder automatically
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
    print("\nERROR: Dataset tidak dijumpai!")
    print("\nPastikan file CSV berada di:")
    print(DATA_PATH)
    exit()


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nMood class distribution:")
print(df["Mood"].value_counts())


# ============================================================
# 4. SELECT FEATURES
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
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Column berikut tidak dijumpai:")
    
    for column in missing_columns:
        print("-", column)

    print("\nColumn yang ada dalam dataset:")
    print(df.columns.tolist())

    exit()


# ============================================================
# 6. REMOVE MISSING VALUES
# ============================================================

df = df.dropna(
    subset=FEATURES + [TARGET]
)

print("\nDataset after removing missing values:")
print(df.shape)


# ============================================================
# 7. CREATE FEATURES AND TARGET
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
# 8. TRAIN / VALIDATION SPLIT
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
# 9. RANDOM FOREST TREE NUMBER EXPERIMENT
# ============================================================

print("\nTesting different number of trees...")

tree_numbers = [10, 50, 100, 150, 200, 300]

tree_results = []

models = {}


for trees in tree_numbers:

    print(f"\nTraining Random Forest with {trees} trees...")


    rf_model = RandomForestClassifier(

        n_estimators=trees,

        max_depth=18,

        min_samples_leaf=5,

        random_state=42,

        n_jobs=-1,

    )


    rf_model.fit(

        X_train,

        y_train

    )


    train_prediction = rf_model.predict(X_train)

    val_prediction = rf_model.predict(X_val)


    train_accuracy = accuracy_score(

        y_train,

        train_prediction

    )


    val_accuracy = accuracy_score(

        y_val,

        val_prediction

    )


    tree_results.append({

        "Number of Trees": trees,

        "Training Accuracy": train_accuracy,

        "Validation Accuracy": val_accuracy

    })


    models[trees] = rf_model



results_df = pd.DataFrame(tree_results)


print("\nTree Experiment Results:")

print(results_df)



# Select best number of trees based on validation accuracy

best_tree_number = results_df.loc[

    results_df["Validation Accuracy"].idxmax(),

    "Number of Trees"

]


print("\nBest Number of Trees:")

print(best_tree_number)



# Use best model for remaining evaluation

model = models[best_tree_number]

print("\nBest Random Forest model selected!")


# ============================================================
# 10. TRAINING ACCURACY
# ============================================================

train_preds = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_preds
)

print("\nTraining Accuracy:")
print(f"{train_accuracy:.4f}")


# ============================================================
# 11. VALIDATION ACCURACY
# ============================================================

val_preds = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_preds
)

print("\nValidation Accuracy:")
print(f"{val_accuracy:.4f}")


# ============================================================
# 12. CLASSIFICATION REPORT
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
# 13. CONFUSION MATRIX
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
# 14. SAVE CONFUSION MATRIX
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
    "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=150
)

plt.close()

print("\nConfusion matrix saved:")
print(confusion_matrix_path)


# ============================================================
# 15. FEATURE IMPORTANCE
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
# 16. SAVE FEATURE IMPORTANCE GRAPH
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
    "feature_importance.png"
)

plt.savefig(
    feature_importance_path,
    dpi=150
)

plt.close()

print("\nFeature importance saved:")
print(feature_importance_path)


# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "mood_model.pkl"
)

joblib.dump(
    model,
    model_path,
    compress=3
)

print("\nTrained model saved:")
print(model_path)


# ============================================================
# 18. FINAL SUMMARY
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
print("- confusion_matrix.png")
print("- feature_importance.png")
print("- mood_model.pkl")

print("\nTraining completed successfully!")