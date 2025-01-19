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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

# Feeding Patter
MODEL_FEED = joblib.load('feeding_pattern_model.joblib')

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
    health_status = health_status_mapping.get(health_status_encoded, "Unknown")

    return {"health_status": health_status}

class FeedPredictionRequest(BaseModel):
    cattle_breed: str
    health_status: str
    status: str
    feeding_amount_KG_morning: float
    score_morning: float
    feeding_amount_KG_noon: float
    score_noon: float
    feeding_amount_KG_evening: float
    score_evening: float
    travel_distance_per_day_KM: float

encodings = {
    'cattle_breed': {'Australian Milking Zebu': 0, 'Ayrshire': 1, 'Friesian': 2, 'Jersey': 3, 'Lanka White': 4, 'Sahiwal': 5},
    'health_status': {'Healthy': 0, 'Sick': 1},
    'status': {'Breeding': 0, 'Bulls': 1, 'Calves ': 2, 'Heifers ': 3, 'Lactating': 4, 'Pregnant': 5},
    'food_type_morning': {'Coconut Poonac': 0, 'Coconut Poonac, Grass': 1, 'Milk': 2},
    'food_type_noon': {'Coconut Poonac, Grass': 0, 'Grass, Paddy Straw': 1, 'Napier Grass, Guinea grass': 2,
                       'Napier Grass, Guinea grass, Para grass': 3, 'Napier Grass, Guinea grass,Gliricidia': 4,
                       'Paddy Straw, Grass (Chopped)': 5, 'Para grass, Gliricidia': 6},
    'food_type_evening': {'Milk': 0, 'Paddy Straw': 1, 'Paddy Straw, Corn': 2, 'Paddy Straw, Grass': 3, 'Paddy Straw, Legumes': 4}
}

@app.post("/predict_food_type")
async def predict_food_type(request: FeedPredictionRequest):
    encoded_input = [
        encodings['cattle_breed'].get(request.cattle_breed, -1),
        encodings['health_status'].get(request.health_status, -1),
        encodings['status'].get(request.status, -1),
        request.feeding_amount_KG_morning,
        request.score_morning,
        request.feeding_amount_KG_noon,
        request.score_noon,
        request.feeding_amount_KG_evening,
        request.score_evening,
        request.travel_distance_per_day_KM
    ]

    if -1 in encoded_input[:3]:  
        raise HTTPException(status_code=400, detail="Invalid input value(s)")

    encoded_input = np.array(encoded_input).reshape(1, -1)

    try:
        prediction = MODEL_FEED.predict(encoded_input)
        morning_pred, noon_pred, evening_pred = prediction[0].split('-')
        food_type_morning = [key for key, value in encodings['food_type_morning'].items() if value == int(morning_pred)][0]
        food_type_noon = [key for key, value in encodings['food_type_noon'].items() if value == int(noon_pred)][0]
        food_type_evening = [key for key, value in encodings['food_type_evening'].items() if value == int(evening_pred)][0]

        return {
            'morning': food_type_morning,
            'noon': food_type_noon,
            'evening': food_type_evening
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Feed patern save
class CattleData(BaseModel):
    cattle_name: str
    health_status: str
    status: str
    food_type_morning: str
    feeding_amount_KG_morning: float
    score_morning: int
    food_type_noon: str
    feeding_amount_KG_noon: float
    score_noon: int
    food_type_evening: str
    feeding_amount_KG_evening: float
    score_evening: int
    feed_platform: str
    # feeding_amount_KG_L: float
    travel_distance_per_day_KM: float
    farmers_id: str
    farmer_name: str

# Endpoint to save cattle data
@app.post("/feed-patterns")
async def save_cattle_data(cattle_data: CattleData):
    try:
        # Add the cattle data to Firestore
        feed_patterns_collection = db.collection("feed_patterns")
        new_doc = feed_patterns_collection.document()
        new_doc.set(cattle_data.dict())

        return {"message": "Cattle data successfully saved", "id": new_doc.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving cattle data: {str(e)}")

# Endpoint to retrieve all cattle data
@app.get("/feed-patterns")
async def get_all_cattle_data():
    try:
        feed_patterns_collection = db.collection("feed_patterns")
        docs = feed_patterns_collection.stream()

        cattle_data_list = [doc.to_dict() for doc in docs]

        return {"data": cattle_data_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cattle data: {str(e)}")

# Endpoint to retrieve cattle data by ID
@app.get("/feed-patterns/{cattle_id}")
async def get_cattle_data_by_id(cattle_id: str):
    try:
        feed_patterns_collection = db.collection("feed_patterns")
        doc = feed_patterns_collection.document(cattle_id).get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Cattle data not found")

        return {"data": doc.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cattle data by ID: {str(e)}")

# Endpoint to retrieve cattle data by farmer
@app.get("/feed-patterns/farmer/{farmer_id}")
async def get_cattle_data_by_farmer(farmer_id: str):
    try:
        feed_patterns_collection = db.collection("feed_patterns")
        docs = feed_patterns_collection.where("farmers_id", "==", farmer_id).stream()

        cattle_data_list = [doc.to_dict() for doc in docs]

        if not cattle_data_list:
            raise HTTPException(status_code=404, detail="No cattle data found for the given farmer")

        return {"data": cattle_data_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cattle data by farmer: {str(e)}")


# Breed Related

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


# Breed Information
file_path = 'breed_dataset/cattle_breed_idintification_dataset.xlsx'
df = pd.read_excel(file_path)

# Combine text-heavy fields into a single feature
df['text_data'] = (
    df['Cattle Breed Name'] + " " +
    df['Pedigree/Lineage'] + " " +
    df['Optimal Rearing Conditions'] + " " +
    df['Physical Characteristics'] + " " +
    df['Temperament'] + " " +
    df['Productivity Metrics']
)

# # Target variable
# target = 'Origin'

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     df['text_data'],
#     df[target],
#     test_size=0.2,
#     random_state=42
# )

# # Vectorize text data
# tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
# X_train_tfidf = tfidf.fit_transform(X_train)
# X_test_tfidf = tfidf.transform(X_test)

# # Train a Naive Bayes classifier
# model = MultinomialNB()
# model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
model = joblib.load('cattle_breed_nlp_model.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')

# Define request model
class InsightRequest(BaseModel):
    breed_name: str
    adopted: bool

# Define output model for FastAPI
class InsightResponse(BaseModel):
    Breed: str
    Adopted: str
    Predicted_Origin: str
    Rearing_Conditions: str
    Temperament: str
    Milk_Production: int
    Lifespan: int

@app.post("/insights", response_model=InsightResponse)
def provide_insights(request: InsightRequest):
    breed_name = request.breed_name
    adopted = request.adopted

    # Filter dataset for the given breed
    breed_data = df[df['Cattle Breed Name'].str.contains(breed_name, case=False, na=False)]

    if breed_data.empty:
        return {"error": "Breed not found in the dataset."}

    # Generate prediction using the model (use text_data column)
    breed_text = breed_data['text_data'].iloc[0]
    breed_tfidf = tfidf.transform([breed_text])
    predicted_origin = model.predict(breed_tfidf)[0]

    # Convert numpy types to native Python types
    insights = InsightResponse(
        Breed=breed_name,
        Adopted="Yes" if adopted else "No",
        Predicted_Origin=predicted_origin,
        Rearing_Conditions=breed_data['Optimal Rearing Conditions'].iloc[0],
        Temperament=breed_data['Temperament'].iloc[0],
        Milk_Production=int(breed_data['Milk Production Ability (Liters/Year)'].iloc[0]),
        Lifespan=int(breed_data['Lifespan (Years)'].iloc[0])
    )
    return insights


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

# Farm Location / border save
class FarmBorderRequest(BaseModel):
    user: str
    farm_name: str
    details: str
    border: list[dict]

@app.post("/mark-farm-border")
async def mark_farm_border(request: FarmBorderRequest):
    try:
        # Prepare farm data
        farm_data = {
            "user": request.user,
            "farm_name": request.farm_name,
            "details": request.details,
            "border": request.border,
        }
        
        # Save to Firestore
        farms_collection = db.collection("farms")
        new_farm_doc = farms_collection.document()
        new_farm_doc.set(farm_data)
        
        return {"message": "Farm border successfully saved", "farm_id": new_farm_doc.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving farm border: {str(e)}")
