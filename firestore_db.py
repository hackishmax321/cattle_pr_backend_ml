import os
from google.cloud import firestore

# Set the environment variable for the service account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cattletracking-d9d96-firebase-adminsdk.json"

# Initialize Firestore DB
db = firestore.Client()

def get_firestore_client():
    return db