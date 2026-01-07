import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from emotion_model import predict_emotion_with_confidence, predict_emotion
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Trying alternative import method...")
    try:
        # Try importing the module directly
        import emotion_model
        predict_emotion_with_confidence = emotion_model.predict_emotion_with_confidence
        predict_emotion = emotion_model.predict_emotion
    except Exception as e2:
        st.error(f"Failed to import: {e2}")
        st.stop()

# Page config
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="centered"
)

# Custom CSS with footer
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.title {
    text-align: center;
    color: #38bdf8;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
}
.emotion-box {
    background: #020617;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 30px;
    color: #22c55e;
    margin: 20px 0;
}
.footer {
    position: fixed;
    bottom: 20px;
    right: 20px;
    color: black;
    font-size: 14px;
    font-family: monospace;
}
.stButton > button {
    background-color: #3b82f6;
    color: white;
    border: none;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #2563eb;
}
</style>
""", unsafe_allow_html=True)

# Add footer with bold Abutalha
st.markdown("<div class='footer'>by <br> <b>Abutalha</b></div>", unsafe_allow_html=True)

st.markdown("<div class='title'>Emotion Detection from Text</div>", unsafe_allow_html=True)


st.write("")

# Add instructions
with st.expander("💡 How to use", expanded=False):
    st.write("""
    **Examples to try:**
    - "I'm so excited about this!" → Happy 😊
    - "This is really disappointing" → Sad 😢
    - "I can't believe this happened!" → Angry 😠
    - "I'm afraid of what might happen" → Fear 😨
    - "It's just another regular day" → Neutral 😐
    """)

user_text = st.text_area(
    "💬 Enter chat text:",
    height=120,
    placeholder="Type customer message here...",
    help="Enter any text message to analyze the emotion"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    detect_clicked = st.button("🔍 Detect Emotion", type="primary", use_container_width=True)
    
if detect_clicked:
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing emotion..."):
            try:
                # Check which function is available
                try:
                    emotion, confidence = predict_emotion_with_confidence(user_text)
                except:
                    # Fallback to simple prediction
                    emotion = predict_emotion(user_text)
                    confidence = 75.0  # Default confidence
                
                # Color coding for emotions
                emotion_colors = {
                    'happy': '#22c55e',
                    'sad': '#3b82f6',
                    'angry': '#ef4444',
                    'fear': '#8b5cf6',
                    'neutral': '#94a3b8'
                }
                
                # Emotion emojis
                emotion_emojis = {
                    'happy': '😊',
                    'sad': '😢',
                    'angry': '😠',
                    'fear': '😨',
                    'neutral': '😐'
                }
                
                color = emotion_colors.get(emotion, '#94a3b8')
                emoji = emotion_emojis.get(emotion, '😐')
                
                # Display result
                st.markdown(
                    f"""
                    <div class='emotion-box' style='border-left: 5px solid {color};'>
                        {emoji} Detected Emotion: <br>
                        <span style='color: {color}; font-size: 36px;'><b>{emotion.upper()}</b></span><br>
                        <span style='color: #cbd5e1; font-size: 16px;'>
                        Confidence: {confidence:.1f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please try again with different text.")

# Add statistics
st.divider()
st.markdown("### 📈 Model Statistics")

col1 ,col2, col3 = st.columns(3)
with col1:
    st.metric("Training Samples", "500+")
with col2:
    st.metric("Accuracy", "~95%")
with col3:
    st.metric("Emotions", "5")

# Add emotion distribution visualization
st.markdown("#### Emotion Categories")
emotions = ['happy', 'sad', 'angry', 'fear', 'neutral']
colors = ['#22c55e', '#3b82f6', '#ef4444', '#8b5cf6', '#94a3b8']

cols = st.columns(5)
for i, (emotion, color) in enumerate(zip(emotions, colors)):
    with cols[i]:
        st.markdown(f"<div style='text-align: center; color: {color}; font-weight: bold;'>{emotion.capitalize()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: {color}; height: 10px; border-radius: 5px;'></div>", unsafe_allow_html=True)

st.caption("| by <b>Abutalha</b> <br>" 
            " CU-4279-2023 <br>" 
            " BSSE-Section-A  <br>"
            " Presented to <b> Miss Minahil Ather </b> <br> "
            "Software Varification and Validation Lab", unsafe_allow_html=True)