import pandas as pd
from sklearn.model_selection import train_test_split
from RandomForest import RandomForest
import numpy as np

dataset = pd.read_csv("C:\\Users\\visha\\OneDrive\\Desktop\\heart.csv")
X = dataset.drop('target', axis=1)
y = dataset["target"]
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForest(n_trees=100)
model.fit(X_train, y_train)

user_input = []
for column in dataset.columns[:-1]:
    while True:
        try:
            value = float(input(f"Enter the value for {column}: "))
            user_input.append(value)
            break
        except ValueError:
            print("Please enter a valid numerical value.")

user_prediction = model.predict([user_input])

if user_prediction[0] == 1:
    print("The model predicts that the user has heart disease.")
    unsure = input("Are you still unsure? Enter 'yes' if you want to proceed to symptoms analysis: ")

    while unsure.lower() != "yes":
        unsure = input("Please enter 'yes' if you want to proceed to symptoms analysis: ")

    print("Let's proceed to symptoms analysis.")
    symptoms_input = []

    # Ask questions related to symptoms analysis
    symptoms_questions = [
        "Chest Pain or Discomfort",
        "Shortness of Breath",
        "Fatigue",
        "Palpitations",
        "Dizziness or Fainting",
        "Nausea",
        "Sweating",
        "Pain in Other Areas",
        "Heartburn or Indigestion",
        "Swelling in the Legs, Ankles, or Feet",
        "Persistent Coughing",
        "Appetite Changes",
        "Anxiety or Restlessness",
        "Difficulty Sleeping",
        "Lack of Stamina",
        "Sudden Weight Gain",
        "Rapid or Irregular Heartbeat",
        "Feeling of Fullness or Pressure in the Chest",
        "Pain that Radiates to the Back, Neck, Jaw, or Shoulder",
        "General Weakness",
        "Lightheadedness",
        "Persistent Indigestion",
        "Flu-like Symptoms, including Body Aches",
        "Feeling of Impending Doom",
        "Bluish Lips or Fingernails (Sign of severe cases)",
        "Confusion or Cognitive Changes",
        "Heavy Sweating",
        "Pain or Discomfort Between Shoulder Blades",
        "Unexplained Weakness or Fatigue",
        "Rapid Breathing",
        "Cold or Clammy Skin",
        "Changes in Skin Color (Pale or Gray)",
        "Reduced Ability to Exercise",
        "Persistent Nausea or Vomiting",
        "Pain or Discomfort in the Upper Abdomen",
        "Unexplained Anxiety",
        "Feeling of a 'Pounding' Heart",
        "Difficulty Swallowing",
        "Increased Heart Rate with Minimal Exertion",
        "Loss of Consciousness",
        "Swollen or Tender Abdomen",
        "Excessive Thirst",
        "Frequent Urination"
    ]
    print(len(symptoms_questions))

    for question in symptoms_questions:
        while True:
            response = input(f"Do you experience {question}? (Enter 'yes' or 'no'): ")
            if response.lower() in ['yes', 'no']:
                symptoms_input.append(response.lower() == "yes")
                break
            else:
                print("Please enter 'yes' or 'no'.")

    
    if sum(symptoms_input) >= 15:
        print("Based on symptoms analysis, you may have heart disease. Please consult a healthcare professional.")
    else:
        print("Based on symptoms analysis, you may not have heart disease. However, consult a healthcare professional for confirmation.")
else:
    print("The model predicts that the user does not have heart disease.")