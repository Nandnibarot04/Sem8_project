from flask import Flask, request, jsonify, render_template
import joblib
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from asgiref.wsgi import WsgiToAsgi  # Convert WSGI to ASGI

app = Flask(__name__, static_folder="static", template_folder="templates")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
logistic_model = joblib.load("D:/Sem 8/NLP_LLM/NLP_Project/logistic_regression_model.pkl")

# Load the saved label encoder if you have it
label_encoder = joblib.load("D:/Sem 8/NLP_LLM/NLP_Project/label_encoder.pkl")

# Create mapping from LABEL_X to original category names
label_mapping = {f"LABEL_{i}": class_name for i, class_name in enumerate(label_encoder.classes_)}

# Load SVM model and ensure it's not a tuple
svm_model_data = joblib.load("D:/Sem 8/NLP_LLM/NLP_Project/svm_model.pkl")
if isinstance(svm_model_data, tuple):
    svm_model, vectorizer = svm_model_data  # Unpack model & vectorizer
else:
    svm_model = svm_model_data  # Single model case

# Load fine-tuned RoBERTa model on GPU
tokenizer = RobertaTokenizer.from_pretrained("D:/Sem 8/NLP_LLM/NLP_Project/roberta_finetuned_model_save")
roberta_model = RobertaForSequenceClassification.from_pretrained("D:/Sem 8/NLP_LLM/NLP_Project/roberta_finetuned_model_save").to(device)

# Extract category labels directly from RoBERTa model
if hasattr(roberta_model.config, "id2label"):
    categories = [roberta_model.config.id2label[i] for i in sorted(roberta_model.config.id2label.keys())]
else:
    categories = [f"Category {i}" for i in range(roberta_model.config.num_labels)]  # Fallback labels
    
# Load the Logistic Regression vectorizer
try:
    logistic_vectorizer = joblib.load("D:/Sem 8/NLP_LLM/NLP_Project/logistic_regression_vectorizer.pkl")
  
except Exception as e:
    print(f"Error loading logistic vectorizer: {e}")
    # Handle error appropriately

def predict_logistic(text):
    try:
        # Transform the text into a feature vector first
        transformed_text = logistic_vectorizer.transform([text])
        # Now predict using the transformed text
        return logistic_model.predict(transformed_text)[0]
    except Exception as e:
        print(f"Error in logistic prediction: {e}")
        return "Error: Could not make prediction"


# Load the SVM vectorizer
try:
    svm_vectorizer = joblib.load("D:/Sem 8/NLP_LLM/NLP_Project/svm_vectorizer.pkl")
except:
    # If not saved separately, you might need to use the same vectorizer as logistic regression
    # or find another solution
    pass

def predict_svm(text):
    try:
        # Transform the text into a feature vector first
        transformed_text = svm_vectorizer.transform([text])
        # Now predict using the transformed text
        return svm_model.predict(transformed_text)[0]
    except Exception as e:
        print(f"Error in SVM prediction: {e}")
        return "Error: Could not make prediction"

def classify_headline(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    # Model prediction
    with torch.no_grad():
        outputs = roberta_model(**inputs)

    # Apply softmax to get probabilities
    probs = F.softmax(outputs.logits, dim=-1)
    
    # Get the highest probability and its corresponding class index
    confidence, prediction = torch.max(probs, dim=-1)
    
    raw_label = roberta_model.config.id2label[prediction.item()]
    
    # Map the raw label to the original category name
    if raw_label in label_mapping:
        predicted_category = label_mapping[raw_label]
    else:
        predicted_category = raw_label  # Fallback to the raw label
    
    return predicted_category, confidence.item()


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Convert Flask app to ASGI
asgi_app = WsgiToAsgi(app)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("headline", "")
    model_type = data.get("model", "logistic")
    
    if model_type == "logistic":
        prediction = predict_logistic(text)
    elif model_type == "svm":
        prediction = predict_svm(text)
    elif model_type == "roberta":
        prediction, confidence = classify_headline(text)
    else:
        return jsonify({"error": "Invalid model selection"}), 400  
    
    return jsonify({"category": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)


