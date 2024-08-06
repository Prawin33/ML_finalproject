from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the model and tokenizer
model = tf.keras.models.load_model(r'C:\Users\Prave\Downloads\ML_Final Project\sentiment_model.h5')
with open(r'C:\Users\Prave\Downloads\ML_Final Project\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
