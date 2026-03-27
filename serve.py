from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class Features(BaseModel):
    age: float
    daily_screen_time_hours: float
    sleep_hours: float

@app.post("/predict")
def predict(data: Features):
    features = [data.age, data.daily_screen_time_hours, data.sleep_hours]
    prediction = model.predict([features])
    return {"addiction_level": prediction[0]}
    pred_label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]  # e.g. [0.2,0.5,0.3]
    class_labels = model.classes_.tolist()  # ['low', 'medium', 'high'] ideally
    
    # pick predicted class probability
    predicted_index = list(class_labels).index(pred_label)
    predicted_prob = float(proba[predicted_index])

    return {
        "addiction_risk": str(pred_label),
        "dependency_probability": round(predicted_prob, 4),
        "probabilities": {
            class_labels[i]: round(float(proba[i]), 4)
            for i in range(len(class_labels))
        },
    }