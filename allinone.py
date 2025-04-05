import streamlit as st
import google.generativeai as genai
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import easyocr
import torch
import pyttsx3
import speech_recognition as sr
from datetime import datetime, timedelta
import time

# Configure Gemini API
genai.configure(api_key="AIzaSyCc4B5Og2hOxnERFBSp95iQ9aT-urSCKM8")  # Replace with your API key

# Set page config
st.set_page_config(
    page_title="🍳 AI Kitchen Assistant",
    page_icon="🍳",
    layout="wide"
)

# Initialize session state variables
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Recipe Simplifier"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Common Components ---
def show_footer():
    st.markdown("---")
    st.caption("🍳 AI Kitchen Assistant v1.0 | Made BY  Muhammad Tarique")

# --- Recipe Simplifier Tab ---
def recipe_simplifier_tab():
    st.title("🥘 Recipe Summarizer & Simplifier")
    st.caption("Paste any long recipe and get a quick, simplified version powered by Gemini AI.")

    recipe_input = st.text_area("📄 Paste your recipe here:", height=300, 
                               placeholder="e.g. 1. Preheat oven to 350°F. 2. Mix flour and sugar...")

    if st.button("✨ Simplify Recipe"):
        if recipe_input.strip() == "":
            st.warning("Please enter a recipe to summarize.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    model = genai.GenerativeModel("gemini-1.5-pro-latest")
                    prompt = f"Summarize and simplify the following recipe for easy understanding:\n\n{recipe_input}"
                    response = model.generate_content(prompt)
                    summarized = response.text.strip()

                    st.subheader("🧾 Simplified Recipe")
                    st.success(summarized)

                except Exception as e:
                    st.error(f"Error: {str(e)[:200]}")

# --- Food Processor Tab ---
@st.cache_resource
def load_ocr_reader():
    try:
        gpu_available = torch.cuda.is_available()
        return easyocr.Reader(['en'], gpu=gpu_available)
    except Exception as e:
        st.error(f"OCR initialization failed: {str(e)}")
        return None

def detect_objects(image, mode):
    """Mock object detection for food/ingredients/bills"""
    height, width = image.shape[:2]
    if mode == "Food Items":
        return [{"label": "Apple", "confidence": 0.92, "bbox": [width//4, height//4, 100, 100]}]
    elif mode == "Ingredients":
        return [{"label": "Flour", "confidence": 0.85, "bbox": [width//2, height//2, 90, 90]}]
    else:  # Receipts
        return [{"label": "Total", "confidence": 0.95, "bbox": [width//6, height//6, 200, 50]}]

def perform_ocr(image_array):
    """Real OCR implementation using EasyOCR"""
    try:
        img_bgr = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
        
        if 'reader' not in st.session_state:
            st.session_state.reader = load_ocr_reader()
        
        if st.session_state.reader is None:
            return "OCR engine failed to initialize"
        
        results = st.session_state.reader.readtext(img_bgr)
        return "\n".join([f"{text} ({confidence:.0%})" for (_, text, confidence) in results])
    
    except Exception as e:
        st.error(f"OCR processing error: {str(e)}")
        return ""

def extract_insights(text):
    """Extract key info from OCR text"""
    insights = []
    text_lower = text.lower()
    
    if "total" in text_lower:
        insights.append("💰 Total amount detected")
    if any(word in text_lower for word in ["expiry", "date"]):
        insights.append("📅 Expiry date mentioned")
    return insights or ["🔍 No key insights automatically detected"]

def food_processor_tab():
    st.title("🍏 Food & Ingredients Bill Processor")
    st.markdown("Upload food/ingredient images or receipts for AI analysis and OCR extraction")

    with st.sidebar:
        st.header("⚙️ Settings")
        detection_mode = st.selectbox(
            "Detection Focus",
            ["Food Items", "Ingredients", "Receipts"]
        )

    uploaded_file = st.file_uploader(
        "📤 Upload Food/Receipt Image (JPG/PNG)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_column_width=True)
            
            st.subheader("🧠 AI Description")
            try:
                model = genai.GenerativeModel("gemini-1.5-pro-latest")
                prompt = f"Describe this food/receipt image in detail: {perform_ocr(img_array)}"
                response = model.generate_content(prompt)
                st.info(response.text)
            except Exception as e:
                st.error(f"AI description failed: {str(e)}")
        
        with col2:
            st.subheader("🔍 Detected Items")
            detections = detect_objects(img_array, detection_mode)
            
            if detections:
                fig, ax = plt.subplots()
                ax.imshow(image)
                for obj in detections:
                    x,y,w,h = obj["bbox"]
                    rect = patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='red',facecolor='none')
                    ax.add_patch(rect)
                    plt.text(x,y,f"{obj['label']} ({obj['confidence']:.0%})",
                            color='red', bbox=dict(facecolor='white', alpha=0.7))
                plt.axis('off')
                st.pyplot(fig)
            else:
                st.warning("No items detected")
            
            st.subheader("💡 Key Insights")
            text = perform_ocr(img_array)
            for insight in extract_insights(text):
                st.success(insight)
        
        with st.expander("📝 View Full OCR Text"):
            st.text(text)
    else:
        st.info("Please upload an image to begin analysis")

# --- Multilingual Chatbot Tab ---
class ChatBot:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        self.chat = None
        self.last_request_time = datetime.now()
        self.request_delay = timedelta(seconds=3)
        self.setup_bot()
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Set Hindi voice if available
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'hindi' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

    def setup_bot(self):
        if "chat" not in st.session_state:
            st.session_state.chat = self.model.start_chat(history=[])
        self.chat = st.session_state.chat

    def enforce_rate_limit(self):
        elapsed = datetime.now() - self.last_request_time
        if elapsed < self.request_delay:
            wait_time = (self.request_delay - elapsed).total_seconds()
            time.sleep(wait_time)
        self.last_request_time = datetime.now()

    def get_response(self, user_input):
        try:
            self.enforce_rate_limit()
            response = self.chat.send_message(
                user_input,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500
                )
            )
            return response.text
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                return "I've reached my usage limit. Please try again later."
            return f"I'm having trouble responding. Error: {str(e)[:100]}..."
    
    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
    
    def listen(self):
        with sr.Microphone() as source:
            st.info("Listening... Speak now")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except Exception as e:
                st.error(f"Could not understand audio: {e}")
                return None

def chatbot_tab():
    st.title("🎙️ Kitchen Assistant ChatBot")
    st.caption("Chat in English or Hindi about cooking, recipes, or food. Say 'quit' to end conversation")
    
    bot = ChatBot()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", 
                                        "content": "Hello! I'm your kitchen assistant. How can I help with cooking today?"})
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.button("🎤 Speak to Assistant"):
        user_input = bot.listen()
        if user_input:
            process_chat_input(user_input, bot)
    
    if prompt := st.chat_input("Ask me about cooking, recipes, or food..."):
        process_chat_input(prompt, bot)

def process_chat_input(prompt, bot):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if prompt.lower() in ['quit', 'exit', 'bye', 'बंद करो', 'अलविदा']:
        response = "Goodbye! Happy cooking! / अलविदा! खाना बनाने का आनंद लें!"
        st.session_state.messages.append({"role": "assistant", "content": response})
        bot.speak(response)
        st.rerun()
    else:
        with st.spinner("Thinking..."):
            response = bot.get_response(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        with st.chat_message("assistant"):
            st.markdown(response)
        
        bot.speak(response)
        st.rerun()

# --- Main App Structure ---
def main():
    st.sidebar.title("🍳 Navigation")
    tabs = {
        "Recipe Simplifier": recipe_simplifier_tab,
        "Food Processor": food_processor_tab,
        "Kitchen ChatBot": chatbot_tab
    }
    
    selected_tab = st.sidebar.radio("Go to", list(tabs.keys()))
    st.session_state.active_tab = selected_tab
    
    tabs[selected_tab]()
    show_footer()

if __name__ == "__main__":
    main()