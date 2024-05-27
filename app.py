import random
import json
import nltk
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import pymssql
from spellchecker import SpellChecker

app = Flask(__name__)

# Load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Initialize SpellChecker
spell = SpellChecker()

# Load intents data
with open("intents.json") as file:
    intents = json.load(file)

# Load Keras model and pickle files
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")

# Define database connection parameters
server = '192.168.180.22'
database = 'WMS-OJT'
username = 'OJT'
password = '123!@#qwe'

# Function to establish database connection
def get_db_connection():
    try:
        conn = pymssql.connect(server=server, user=username, password=password, database=database, timeout=30)
        cursor = conn.cursor()
        print("Database connection successful")
        return conn, cursor
    except Exception as e:
        print("Error connecting to database:", e)
        return None, None

conn, cursor = get_db_connection()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    ints = predict_class(msg)
    res = get_response(ints, msg)
    return res

# Function to execute database query
def execute_query(query):
    global conn, cursor
    try:
        # Attempt to execute the query
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except (pymssql.OperationalError, pymssql.InterfaceError):
        # If a connection error occurs, try to reconnect
        print("Reconnecting to database...")
        conn, cursor = get_db_connection()
        if conn and cursor:
            try:
                cursor.execute(query)
                result = cursor.fetchall()
                return result
            except Exception as e:
                print("Error executing query after reconnecting:", e)
                return None
        else:
            return None
    except Exception as e:
        print("Error executing query:", e)
        return None

def clean_up_sentence(sentence):
    try:
        sentence_words = nltk.word_tokenize(sentence)
    except Exception as e:
        print("Error tokenizing sentence:", e)
        return []

    # Correct spelling errors
    sentence_words = [spell.correction(word.lower()) if word.islower() else word for word in sentence_words]
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word.islower()]  
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = bow(sentence, words)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, msg):
    tag = ints[0]["intent"]
    list_of_intents = intents["intents"]
    for intent in list_of_intents:
        if intent["tag"] == tag:
            if tag == "total_storage":
                storage_types = ["STORAGECOLD", "STORAGEDRY", "STORAGECHILLER", "STORAGEAIRCON"]
                response = ""
                # Retrieve counts for the specified storage type and direction
                for storage_type in storage_types:
                    if storage_type.lower() in msg.lower():
                        if "outbound" in msg.lower():
                            query_outbound = f"SELECT COUNT(*) FROM WMS.Outbound WHERE StorageType = '{storage_type}'"
                            result_outbound = execute_query(query_outbound)
                            count_outbound = result_outbound[0][0] if result_outbound else 0
                            response = f"For {storage_type} storage:\nOutbound count: {count_outbound}"
                            return response
                        elif "inbound" in msg.lower():
                            query_inbound = f"SELECT COUNT(*) FROM WMS.Inbound WHERE StorageType = '{storage_type}'"
                            result_inbound = execute_query(query_inbound)
                            count_inbound = result_inbound[0][0] if result_inbound else 0
                            response = f"For {storage_type} storage:\nInbound count: {count_inbound}"
                            return response
                # If no specific storage type is mentioned or direction is not clear, return a generic response
                return "Please specify both the storage type and direction (inbound/outbound)."
            
            elif tag == "total_inbound_outbound":
                categories = ["Arrival", "Departure", "StartUnload", "CompleteUnload", "ICNTotalQty", "ArrivalTime", "DepartureTime", "StartLoading", "CompleteLoading", "TotalQty", "StorageType", "Supplier", "SubmittedBy", "PostedBy", "ApprovedBy", "RejectBy", "CheckedBy", "CancelledBy", "DocumentBy", "PutAwayBy", "AddedBy", "LastEditedBy", "AcceptBy", "RFPutAwayBy", "AuthorizedBy", "SuppliedBy", "RFCheckBy"]
                for category in categories:
                    if category.lower() in msg.lower():
                        if category in ["Arrival", "Departure", "StartUnload", "CompleteUnload", "ICNTotalQty", "StorageType", "Supplier", "SubmittedBy", "PostedBy", "ApprovedBy", "RejectBy", "CheckedBy", "CancelledBy", "DocumentBy", "PutAwayBy", "AddedBy", "LastEditedBy", "AcceptBy", "RFPutAwayBy", "AuthorizedBy"]:
                            if "outbound" in msg.lower():
                                query = f"SELECT COUNT(*) FROM WMS.Outbound WHERE {category} IS NOT NULL"
                            elif "inbound" in msg.lower():
                                query = f"SELECT COUNT(*) FROM WMS.Inbound WHERE {category} IS NOT NULL"
                        elif category in ["ArrivalTime", "DepartureTime", "StartLoading", "CompleteLoading", "TotalQty", "StorageType", "AddedBy", "LastEditedBy", "SubmittedBy", "PostedBy", "RejectBy", "RFCheckBy", "CheckedBy", "SuppliedBy", "CancelledBy"]:
                            query = f"SELECT COUNT(*) FROM WMS.Outbound WHERE {category} IS NOT NULL"
                        result = execute_query(query)
                        total_count = result[0][0] if result else "N/A"
                        response = random.choice(intent["responses"]).format(CATEGORY=category, TOTAL_COUNT=total_count)
                        return response
            elif tag == "facility_location":
                location = [word for word in msg.split() if word.lower() not in ["where", "is", "located", "specifically", "in"]]
                response = get_facility_location(location)
                return response
            else:
                response = random.choice(intent["responses"])
                return response
    return "I'm sorry, I didn't understand that."

def get_facility_location(location):
    response = "I'm sorry, I couldn't find any information related to your query. Please provide specific questions related to Mets Logistics."
    location_str = " ".join(location)

    if "cavite" in location_str.lower():
        response = "Mets Logistics has facilities in Carmona, Banahaw, and Maguyam in Cavite."

    if "cebu" in location_str.lower():
        response = "Mets Logistics has facilities in Mandaue, Cebu City."

    if "cagayan de oro" in location_str.lower():
        response = "Mets Logistics has facilities in Tablon, Cagayan de Oro City."

    return response

if __name__ == "__main__":
    app.debug = True
    app.run()
