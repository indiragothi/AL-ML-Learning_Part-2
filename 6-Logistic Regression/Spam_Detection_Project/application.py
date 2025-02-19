from flask import Flask, render_template, request
import joblib

# Initialize Flask app
application = Flask(__name__)
app=application

# Load the vectorizer and model
vectorizer = joblib.load('models/vectorizer.pkl')
model = joblib.load('models/logistic_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input text from the form
        message = request.form['message']
        
        # Preprocess and vectorize the input
        data = vectorizer.transform([message])
        
        # Predict using the model
        prediction = model.predict(data)[0]
        
        # Map the numeric prediction to "Spam" or "Ham"
        result = "Spam" if prediction == 1 else "Ham"
        
        return render_template('index.html', prediction=result, message=message)

if __name__=="__main__":
    app.run(host="0.0.0.0")