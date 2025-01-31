import requests

url = "http://localhost:5000/predict"

data = {
    "age": "29.0",
    "gender" : "F",
    "height_cm" : "72.3",
    "weight_kg" : "75.24",
    "body_fat_%" : "21.3", 
    "diastolic" : "80.0",
    "systolic" : "130.0",
    "gripForce" : "54.9",
    "sit_and_bend_forward_cm" : "18.4",
    "sit_ups_counts" : "60.0",
    "broad_jump_cm" : "217.0"
}

response = requests.post(url, json=data).json()
print(response)