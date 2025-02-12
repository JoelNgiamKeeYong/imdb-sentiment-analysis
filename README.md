# 🎬 IMDB Movie Sentiment Analysis

## 🚀 Business Scenario

Understanding public sentiment towards movies is crucial for studios, distributors, and even moviegoers. Positive reviews can drive box office success, while negative reviews can significantly impact a film's reception. This project develops a deep learning model to analyze movie reviews from the IMDB dataset and predict their sentiment (positive or negative). This information can be invaluable for:

- **Movie studios:** Gauging audience reactions to test screenings or early releases.
- **Distributors:** Understanding how a film is being received by critics and the public.
- **Moviegoers:** Making informed decisions about which movies to watch.

---

## 🧠 Business Problem

Analyzing movie review sentiment manually is time-consuming and challenging due to the sheer volume of reviews. An automated sentiment analysis model can help:

- **Quickly process large datasets** of reviews.
- **Identify trends** in public opinion about movies.
- **Provide insights** into specific aspects of films that resonate with audiences.

---

## 🛠️ Solution Approach

This project uses a Recurrent Neural Network (RNN), specifically LSTMs (Long Short-Term Memory networks), for sentiment analysis. The workflow includes:

### 1️⃣ **Data Collection and Preprocessing**

- **Dataset Download:** The IMDB movie review dataset was downloaded from Kaggle.
- **Data Cleaning:** HTML tags and extra whitespace were removed from the review text to improve data quality. This is a _crucial_ step, as raw HTML will confuse the model.
- **Text Tokenization:** A `Tokenizer` was used to convert the text reviews into numerical sequences, representing each word with an integer. A vocabulary size of 5000 was used, taking the most frequent 5000 words.
- **Padding and Truncating:** Sequences were padded or truncated to a fixed length (200 words) using `pad_sequences`. This is essential for consistent input to the neural network.
- **Train/Test Split:** The data was split into training (80%) and testing (20%) sets.

### 2️⃣ **Model Building and Training**

- **LSTM Network:** A sequential model was built using an Embedding layer to create word embeddings, an LSTM layer to capture long-range dependencies in the text, and a Dense output layer with a sigmoid activation function for binary classification (positive/negative).
- **Compilation:** The model was compiled using the Adam optimizer, binary cross-entropy loss function, and accuracy as the metric.
- **Training with Early Stopping:** The model was trained on the training data. _Crucially_, early stopping was implemented to prevent overfitting. The training process stops if the validation loss doesn't improve for a certain number of epochs, and the best model weights are restored.

### 3️⃣ **Model Evaluation**

- **Test Set Performance:** The trained model was evaluated on the held-out test set to assess its real-world performance. Loss and accuracy metrics were calculated.
- **Model and Tokenizer Saving:** The trained model and the fitted tokenizer were saved for later use in the Streamlit application.

### 4️⃣ **Streamlit App Development**

- **User Interface:** A Streamlit app was created to provide an interactive way for users to input movie reviews and get sentiment predictions.
- **Model Loading:** The saved model and tokenizer are loaded into the Streamlit app.
- **Prediction:** The app takes user input, preprocesses it (cleaning and tokenizing), and uses the loaded model to predict the sentiment.
- **Results Display:** The app displays the sentiment verdict (positive/negative), confidence score, and some key insights based on the prediction.

---

## 📊 Model Performance (Example - Replace with your actual results)

| Metric        | Value  |
| ------------- | ------ |
| Test Loss     | 0.3313 |
| Test Accuracy | 0.8734 |

---

### 🔖 Key Findings (Example - Replace with your insights)

- The model achieved good accuracy on the test set, demonstrating its ability to learn sentiment from text reviews.
- Longer, more detailed reviews generally lead to more confident predictions.

---

## ⚠️ Limitations

1️⃣ **Vocabulary Size:** The vocabulary was limited to the top 5000 words. Less frequent words are not considered, which might impact performance on reviews with unusual vocabulary.

2️⃣ **Contextual Understanding:** While LSTMs capture some context, they may not perfectly understand complex or nuanced language, sarcasm, or irony.

3️⃣ **Data Bias:** The IMDB dataset might have biases that could affect the model's performance on other types of movie reviews.

---

## 🔄 Key Skills Demonstrated

🔹 **Natural Language Processing (NLP)**
🔹 **Recurrent Neural Networks (RNNs) - LSTMs**
🔹 **Deep Learning Model Development with TensorFlow/Keras**
🔹 **Data Preprocessing for Text Data**
🔹 **Streamlit App Development**
🔹 **Model Evaluation and Saving**

---

## 🛠️ Technical Tools & Libraries

- **Python:** Core programming language.
- **Pandas:** Data handling & preprocessing.
- **NumPy:** Numerical computations.
- **TensorFlow/Keras:** Deep learning framework.
- **scikit-learn:** Data splitting and model evaluation.
- **Streamlit:** Web app framework.
- **Pickle:** Saving and loading Python objects (model and tokenizer).

---

## 🚀 Final Thoughts

This project demonstrates how deep learning can be used to effectively analyze movie review sentiment. The Streamlit app provides a user-friendly way to interact with the model. Future work could include exploring different model architectures, using pre-trained word embeddings, or incorporating more data to further improve performance and robustness.
