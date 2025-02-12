import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Initialize session state for model and tokenizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with proper error handling"""
    try:
        # Check if files exist
        if not os.path.exists('sentiment_model.keras'):
            raise FileNotFoundError("Model file 'sentiment_model.keras' not found")
        if not os.path.exists('tokenizer.pickle'):
            raise FileNotFoundError("Tokenizer file 'tokenizer.pickle' not found")
        
        # Clear backend session
        tf.keras.backend.clear_session()
        
        # Load model
        model = tf.keras.models.load_model('sentiment_model.keras')
        
        # Load tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {str(e)}")
        return None, None

def predict_sentiment(review, model, tokenizer):
    """Make prediction with error handling"""
    try:
        # Convert the review text into a sequence
        sequence = tokenizer.texts_to_sequences([review])
        
        # Pad the sequence
        padded_sequence = pad_sequences(sequence, maxlen=200)
        
        # Get prediction and convert to Python float
        prediction = float(model.predict(padded_sequence, verbose=0)[0][0])
        
        return prediction
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    # Page config
    st.set_page_config(
        page_title="Movie Review Sentiment Analysis",
        page_icon="🎬",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .github-link {
            display: inline-flex;
            align-items: center;
            padding: 8px 16px;
            background-color: #24292e;
            color: white !important;
            text-decoration: none;
            border-radius: 6px;
            font-weight: bold;
            height: 40px;
            margin-top: 8px;
        }
        .github-link:hover {
            background-color: #2f363d;
        }
        .stProgress > div > div > div > div {
            background-color: #ff4b4b;
        }
        .positive .stProgress > div > div > div > div {
            background-color: #00cc00;
        }
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 0;
        }
        .title-container {
            margin-bottom: 0 !important;
        }
        .title-text {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model or tokenizer. Please check if the files exist and try again.")
        return

    # Header section with GitHub link
    st.markdown(
        """
        <div class="header-container">
            <div class="title-container">
                <h1 class="title-text">🎬 Movie Review Sentiment Analysis</h1>
            </div>
            <a href="https://github.com/JoelNgiamKeeYong/imdb-sentiment-analysis" class="github-link">
                <svg style="vertical-align: middle; margin-right: 8px" height="20" width="20" viewBox="0 0 16 16" fill="white">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
                View on GitHub
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.write('Enter your movie review below and let our AI model analyze its sentiment.')

    # Main content in columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Initialize session state for review if it doesn't exist
        if 'review' not in st.session_state:
            st.session_state.review = ""

        # Input text box for review
        review = st.text_area(
            'Movie Review:',
            height=200,
            placeholder="Type or paste your movie review here... The more detailed your review, the better the analysis!",
            value=st.session_state.review,
            help="Enter a detailed movie review. The model works best with reviews that are at least a few sentences long."
        )

        predict_button = st.button('🔍 Analyze Sentiment', type='primary', use_container_width=True)

    with col2:
        if predict_button:
            if not review.strip():
                st.warning('⚠️ Please enter a review to analyze.')
            else:
                with st.spinner('🤔 Analyzing sentiment...'):
                    prediction = predict_sentiment(review, model, tokenizer)
                    
                    if prediction is not None:
                        # Create a card-like container for results
                        results_container = st.container()
                        with results_container:
                            st.write('### 📊 Analysis Results')
                            
                            # Sentiment verdict with emoji
                            if prediction > 0.5:
                                st.success('📈 Verdict: POSITIVE')
                                sentiment_class = "positive"
                            else:
                                st.error('📉 Verdict: NEGATIVE')
                                sentiment_class = "negative"
                            
                            # Confidence score
                            confidence = max(prediction, 1-prediction)
                            st.markdown(f"**Confidence Score:** {confidence:.1%}")
                            
                            # Sentiment meter
                            st.markdown(f'<div class="{sentiment_class}">', unsafe_allow_html=True)
                            st.progress(float(prediction))
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Additional insights
                            st.write('### 🔍 Key Insights')
                            if prediction > 0.8:
                                st.write("This review is overwhelmingly positive!")
                            elif prediction > 0.6:
                                st.write("This review leans positive.")
                            elif prediction < 0.2:
                                st.write("This review is strongly negative.")
                            elif prediction < 0.4:
                                st.write("This review leans negative.")
                            else:
                                st.write("This review has mixed sentiments.")

    # Information section at the bottom
    st.divider()
    with st.expander("ℹ️ About this App", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How it Works
            This application uses a sophisticated deep learning model (LSTM) trained on the IMDB movie reviews dataset. 
            The model processes your review through these steps:
            1. Text preprocessing
            2. Converting words to numerical sequences
            3. Neural network analysis
            4. Sentiment prediction
            """)
            
        with col2:
            st.markdown("""
            ### Tips for Best Results
            - Write detailed reviews (at least 2-3 sentences)
            - Use proper punctuation and grammar
            - Be specific about what you liked or disliked
            - Include specific aspects of the movie (acting, plot, etc.)
            """)

if __name__ == "__main__":
    main()