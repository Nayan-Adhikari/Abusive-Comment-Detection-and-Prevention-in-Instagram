"""
Live Hinglish Comment Testing
Project: Abusive Comment Detection and Prevention
"""

import joblib

MODEL_PATH = "models/ensemble_tfidf.joblib"  # or baseline_tfidf_lr.joblib

model = joblib.load(MODEL_PATH)

print("\n🔹 Live Abusive Comment Detection 🔹")
print("Type a Hinglish comment and press Enter")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter comment: ")
    if text.lower() == "exit":
        print("Exiting...")
        break

    prediction = model.predict([text])[0]
    print(f"Prediction → {prediction}")

    if prediction == "safe":
        print("Action: ✅ No action required\n")
    elif prediction == "warning":
        print("Action: ⚠ Warning issued\n")
    else:
        print("Action: 🚨 Severe abuse detected → Escalate\n")
