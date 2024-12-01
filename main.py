# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import bcrypt
import os
import shutil
from typing import List, Optional
from firestore_db import get_firestore_client
import joblib
import pandas as pd
from google.cloud import firestore
from datetime import datetime
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import uuid
import firebase_admin
from firebase_admin import credentials, db as FireDB
import traceback
from math import radians, sin, cos, sqrt, atan2

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load MOdels Health
MODEL_HEALTH = joblib.load('cow_health_model.joblib')

# Growth Model
MODEL_GROWTH = joblib.load('model_growth_in_weight.joblib')

# Load Breed detection Model
MODEL_BREED = load_model("model_breed_detect.h5")
CLASS_BREED = ['Ayrshire', 'Friesian', 'Jersey', 'Sahiwal', 'Local Lankan (Lankan White)', 'Zebu']

# Load Pest detection Models
MODEL_PESTS = load_model("model_pests_detect.h5")
CLASS_PESTS = ['Mastitis', ' Tick Infestation', 'Dermatophytosis (RINGWORM)', 'Fly Strike (MYIASIS)', 'Foot and Mouth disease', 'Lumpy Skin', 'Black Quarter (BQ)', 'Parasitic Mange']

# Db connection
db = get_firestore_client()
cred = credentials.Certificate("cattletracking-d9d96-firebase-adminsdk.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://cattletracking-d9d96-default-rtdb.firebaseio.com/"  # Your Firebase Realtime Database URL
})

class User(BaseModel):
    username: str
    full_name: str
    email:str
    contact: str
    password: str
    nic: str

class LoginUser(BaseModel):
    username: str
    password: str


users_db = {}

@app.post("/register")
async def register_user(user: User):
    user_ref = db.collection("users").document(user.username)
    if user_ref.get().exists:
        raise HTTPException(status_code=400, detail="Username already registered")

    # Hash the password before storing it
    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())
    user_data = user.dict()
    user_data["password"] = hashed_password.decode('utf-8')

    user_ref.set(user_data)
    return {"message": "User registered successfully", "user": user_data}

@app.post("/login")
async def login_user(user: LoginUser):
    user_ref = db.collection("users").document(user.username)
    user_doc = user_ref.get()

    if not user_doc.exists:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    user_data = user_doc.to_dict()
    
    # Check the hashed password
    if not bcrypt.checkpw(user.password.encode('utf-8'), user_data["password"].encode('utf-8')):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    user_data.pop("password")  # Remove the password field from the response

    return {"message": "Login successful", "user": user_data}


class Animal(BaseModel):
    name: str
    owner: str
    type: str
    dob: str
    gender: str
    milk_ability: bool
    status: str
    health: str
    image: str 

@app.post("/add-animal")
async def add_animal(animal:Animal):
    # Verify owner exists
    owner_ref = db.collection("users").document(animal.owner)
    if not owner_ref.get().exists:
        raise HTTPException(status_code=404, detail="Owner not found")

    # Save the image file
    # image_filename = f"{uuid.uuid4()}_{image.filename}"
    # image_path = f"uploads/{image_filename}"  # Adjust the directory as needed

    # with open(image_path, "wb") as f:
    #     f.write(await image.read())

    # Prepare animal data
    animal_data = {
        "name": animal.name,
        "owner": animal.owner,
        "type": animal.type,
        "dob": animal.dob,
        "gender": animal.gender,
        "milk_ability": animal.milk_ability,
        "status": animal.status,
        "health": animal.health,
        "image": animal.image,
    }

    # Store the animal record in Firestore
    animal_ref = db.collection("animals").document()
    animal_ref.set(animal_data)

    return JSONResponse(
        content={
            "message": "Animal added successfully",
            "animal_id": animal_ref.id,
            "image_path": "image_path",
        },
        status_code=200,
    )

@app.get("/animals/{owner}")
async def get_animals_by_owner(owner: str):
    animals_ref = db.collection("animals").where("owner", "==", owner).stream()
    animals = []
    for animal in animals_ref:
        animal_data = animal.to_dict()
        animal_data["id"] = animal.id  # Add Firestore document ID
        animals.append(animal_data)

    if not animals:
        raise HTTPException(status_code=404, detail="No animals found for this owner")
    return {"owner": owner, "animals": animals}


# Predict Health
class HealthStatusRequest(BaseModel):
    reproductive_status: str
    feeding_amount_KG_1: float
    feeding_amount_KG_2: float
    average_food_weight_KG: float
    travel_distance_per_day_KM: float

health_status_mapping = {0: 'Healthy', 1: 'Sick', 2: 'Underweight'}
reproductive_status_mapping = {'Breeding': 0, 'Lactating': 1, 'Non-reproductive': 2, 'Pregnant': 3}



# Define the endpoint for health status prediction
@app.post("/predict-health-status")
def predict_health_status(request: HealthStatusRequest):
    # Validate and encode the reproductive status
    if request.reproductive_status not in reproductive_status_mapping:
        raise HTTPException(status_code=400, detail=f"Invalid reproductive_status: {request.reproductive_status}")
    reproductive_status_encoded = reproductive_status_mapping[request.reproductive_status]

    # Prepare the input features as a numpy array
    input_features = np.array([[reproductive_status_encoded, 
                                request.feeding_amount_KG_1, 
                                request.feeding_amount_KG_2, 
                                request.average_food_weight_KG, 
                                request.travel_distance_per_day_KM]])

    # Make the prediction
    try:
        health_status_encoded = MODEL_HEALTH.predict(input_features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # Convert the encoded health status back to the readable format
    health_status = health_status_mapping.get(health_status_encoded, "Unknown")

    return {"health_status": health_status}


class Breed(BaseModel):
    image: str

# Predict Cattle Breed
@app.post("/predict-breed")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image using the loaded model.
    """
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess the image
        image = Image.open(file_path).convert("RGB")  # Ensure RGB format
        image = image.resize((48, 48))
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the label
        predictions = MODEL_BREED.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))  # Highest probability

        # Map the predicted index to the class name
        predicted_label = CLASS_BREED[predicted_index]

        # Clean up the uploaded file
        os.remove(file_path)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")
    
# Predict Pests
@app.post("/predict-pest")
async def predict_pest(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded image for pest attack detection.
    """
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Preprocess the image
        image = Image.open(file_path).convert("RGB")  # Ensure RGB format
        image = image.resize((48, 48))  # Resize to model's input size
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the pest attack label
        predictions = MODEL_PESTS.predict(image_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))  # Highest probability

        # Map the predicted index to the pest attack label
        predicted_label = CLASS_PESTS[predicted_index]

        # Clean up the uploaded file
        os.remove(file_path)

        return {
            "predicted_label": predicted_label,
            "confidence": confidence
        }
    except Exception as e:
        os.remove(file_path)  # Clean up the uploaded file in case of error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# Growth Monitor
CATTLE_BREED_ENCODING = {
    ' ': 0,
    'AUSTRALIAN MILKING ZEBU': 1,
    'AYRSHIRE': 2,
    'FRIESIAN': 3,
    'JERSEY': 4,
    'LANKA WHITE': 5,
    'SAHIWAL': 6,
    'nan': 7
}

LACTATION_STAGE_ENCODING = {
    'EARLY': 0,
    'LATE': 1,
    'MID': 2,
    'nan': 3
}

REPRODUCTIVE_STATUS_ENCODING = {
    ' PREGNANT': 0,
    'NOT PREGNANT': 1,
    'PREGNANT': 2,
    'nan': 3
}

# Define input data model
class CattleData(BaseModel):
    cattle_breed: str
    height_cm: float
    age_years: float
    feed_kg_per_day: float
    lactation_stage: str
    reproductive_status: str

@app.post("/predict-growth-weight")
async def predict_weight(data: CattleData):
    
    try:
        # Encode categorical variables
        cattle_breed_encoded = CATTLE_BREED_ENCODING.get(data.cattle_breed, 0)  # Default to 0 if not found
        lactation_stage_encoded = LACTATION_STAGE_ENCODING.get(data.lactation_stage, 3)  # Default to 3 if not found
        reproductive_status_encoded = REPRODUCTIVE_STATUS_ENCODING.get(data.reproductive_status, 3)  # Default to 3 if not found

        # Prepare input for the model
        input_data = np.array([[cattle_breed_encoded, data.height_cm, data.age_years, data.feed_kg_per_day, lactation_stage_encoded, reproductive_status_encoded]])

        # Predict weight using the model
        predicted_weight = MODEL_GROWTH.predict(input_data)
        print(predicted_weight[0])

        return {"predicted_weight": predicted_weight[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# GET IoT loaction
def calculate_duration(loc1, loc2, speed_kmph=10):
    """
    Calculate the straight-line distance and approximate duration between two locations.

    Args:
        loc1 (dict): {"latitude": float, "longitude": float} for the first location.
        loc2 (dict): {"latitude": float, "longitude": float} for the second location.
        speed_kmph (float): Assumed speed in kilometers per hour (default is 50 km/h).

    Returns:
        dict: {"distance_km": float, "duration_minutes": float}
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = radians(loc1["latitude"]), radians(loc1["longitude"])
    lat2, lon2 = radians(loc2["latitude"]), radians(loc2["longitude"])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of the Earth in kilometers
    R = 6371.0
    distance_km = 100 * 24 * R * c

    # Calculate duration (time = distance / speed)
    duration_hours = distance_km / speed_kmph
    duration_minutes = duration_hours * 60 

    return {"distance_km": distance_km, "duration_minutes": duration_minutes}

# Example usage with the provided data
location = {"latitude": 6.039483, "longitude": 80.211142}
location24 = {"latitude": 6.039489, "longitude": 80.211146}

result = calculate_duration(location, location24)
print(f"Distance: {result['distance_km']:.4f} km")
print(f"Duration: {result['duration_minutes']:.2f} minutes")



# Locate Farm and Cattle (IOT)
@app.get("/location")
async def get_location():
    try:
        # Reference to the location node
        ref = FireDB.reference("location")
        ref24 = FireDB.reference("location24")
        location_data = ref.get()
        location24_data = ref24.get()

        print(location_data)

        
        
        if not location_data:
            raise HTTPException(status_code=404, detail="Location data not found")
        
        duration = calculate_duration(location_data, location24_data)
        
        return {"location": location_data, "location_24": location24_data, "duration": duration}
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error fetching location data: {str(e)}")
