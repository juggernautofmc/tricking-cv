import joblib
import pandas as pd

class TrickClassifier:
    def __init__(self, model_path="tricking_classifier.joblib"):
        artifact = joblib.load(model_path)

        self.model = artifact["model"]
        self.features = artifact["features"]

    def predict(self, metrics):
        missing = [
            feature
            for feature in self.features
            if feature not in metrics
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )
        
        x = pd.DataFrame(
            [[metrics[name] for name in self.features]],
            columns=self.features,
        )

        predicted_class = self.model.predict(x)[0]
        probabilities = self.model.predict_proba(x)[0]

        confidence_by_class = {
            class_name: float(probability)
            for class_name, probability in zip(
                self.model.classes_,
                probabilities,
            )
        }

        return {
            "prediction": predicted_class,
            "confidence": max(confidence_by_class.values()),
            "probabilities": confidence_by_class,
        }