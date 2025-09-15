# client.py
import requests

payload = {
  "gender": "Female",
  "age": 54.0,
  "hypertension": 0,
  "heart_disease": 1,
  "smoking_history": "never",
  "bmi": 27.32,
  "HbA1c_level": 6.6,
  "blood_glucose_level": 140
}

url = "http://localhost:8000/score"
r = requests.post(url, json=payload, timeout=20)
print(r.status_code, r.json())
