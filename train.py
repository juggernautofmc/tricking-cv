import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

df = pd.read_csv("dataset.csv")

df = df[df["red_flag"] == False]

metrics = [
    "in_air_frames",
    "max_hip_rotation",
    "takeoff_foot",
    "landing_foot"
]

x = df[metrics]
y = df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=67
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    min_samples_leaf=4,
    random_state=67
)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

scores = cross_val_score(
    model,
    x,
    y,
    cv=5
)

print("Mean Score: ", scores.mean())
print("Train:", model.score(x_train, y_train))
print("Test:", model.score(x_test, y_test))
