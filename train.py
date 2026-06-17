import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, RepeatedStratifiedKFold

df = pd.read_csv("dataset.csv")

df = df[df["red_flag"] == False]

features = [
    "in_air_frames",
    "max_hip_rotation",
    "takeoff_foot",
    "landing_foot",
    "left_ft_ht"
]

x = df[features]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=1
)

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=3,
    random_state=1
)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)


scores = cross_val_score(
    model,
    x,
    y,
    cv=cv
)

artifact = {
    "model": model,
    "features": features
}

joblib.dump(artifact, "tricking_classifier.joblib")


# oof_predictions = cross_val_predict(
#     model,
#     x,
#     y,
#     cv=cv
# )

print("Train:", model.score(x_train, y_train))
print("Test:", model.score(x_test, y_test))

print("Scores:", scores)
print("Mean:", scores.mean())
print("Std:", scores.std())

# print("Classes:", model.classes_)
# print("OOF confusion matrix:")
# print(confusion_matrix(y, oof_predictions))

# print("\nOOF classification report:")
# print(classification_report(y, oof_predictions))
