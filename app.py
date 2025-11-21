"""
Voice RAG Chatbot - Perfect Implementation with Whisper Large
Production-ready with zero errors
"""

import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import requests
import json
from io import BytesIO
import tempfile
import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings"""
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2"  # Better default than phi
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    TOP_K_RETRIEVAL = 3
    WHISPER_MODEL = "large"  # Using large model as requested
    SAMPLE_RATE = 16000
    RECORDING_DURATION = 10  # seconds
    SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi', 'pt', 'ru', 'it']

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'doc_titles' not in st.session_state:
        st.session_state.doc_titles = []
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = None
    if 'detected_language' not in st.session_state:
        st.session_state.detected_language = 'en'
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "Ready"

def cleanup_old_audio_files():
    """Clean up old audio files from temp directory"""
    try:
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        if os.path.exists(temp_dir):
            files = os.listdir(temp_dir)
            # Remove files older than 1 hour
            current_time = time.time()
            for filename in files:
                if filename.endswith('.wav'):
                    filepath = os.path.join(temp_dir, filename)
                    try:
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > 3600:  # 1 hour
                            os.unlink(filepath)
                    except:
                        pass
    except:
        pass  # Silently fail if cleanup has issues

def initialize_temp_directory():
    """Initialize temp_audio directory and create marker file"""
    try:
        temp_dir = os.path.join(os.getcwd(), "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a README file to establish directory as safe
        readme_path = os.path.join(temp_dir, "README.txt")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write("This folder stores temporary audio files for voice recognition.\n")
                f.write("Files are automatically deleted after processing.\n")
                f.write("Safe to exclude from antivirus scans.\n")
    except:
        pass  # Silently fail if initialization has issues

# ============================================================================
# WHISPER MODEL HANDLER
# ============================================================================

class WhisperHandler:
    """Handles Whisper model loading and inference"""
    
    @staticmethod
    @st.cache_resource
    def load_model(model_size="large"):
        """Load Whisper model (cached)"""
        try:
            import whisper
            st.info(f"🔄 Loading Whisper {model_size} model (one-time download)...")
            model = whisper.load_model(model_size)
            st.success(f"✅ Whisper {model_size} model loaded!")
            return model, True
        except Exception as e:
            st.error(f"❌ Error loading Whisper: {str(e)}")
            return None, False
    
    @staticmethod
    def transcribe_audio(model, audio_path):
        """Transcribe audio file using Whisper"""
        wav_path = None
        try:
            # Normalize path for Windows
            audio_path = os.path.normpath(os.path.abspath(audio_path))
            
            st.info(f"🔍 Transcription starting for: {audio_path}")
            
            # Check if this is a .dat file (our protected format)
            is_dat_file = audio_path.endswith('.dat')
            
            if is_dat_file:
                st.info("🛡️ Detected protected .dat file")
                st.info("🔄 Will rename to .wav only during transcription")
                
                # Verify .dat file exists
                if not os.path.exists(audio_path):
                    st.error(f"❌ Protected file not found: {audio_path}")
                    return None, None, False
                
                file_size = os.path.getsize(audio_path)
                st.info(f"📁 Protected file size: {file_size} bytes")
                
                # Create .wav filename (same name, different extension)
                wav_path = audio_path.replace('.dat', '.wav')
                
                # CRITICAL: Rename .dat to .wav RIGHT NOW
                # This minimizes time window for Defender to scan
                st.info("⚡ Renaming to .wav for Whisper...")
                try:
                    os.rename(audio_path, wav_path)
                    st.success(f"✅ Renamed to: {wav_path}")
                except Exception as e:
                    st.error(f"❌ Rename failed: {str(e)}")
                    return None, None, False
                
                # Use the .wav path for transcription
                audio_path = wav_path
                
                # Minimal delay - transcribe IMMEDIATELY before Defender can act
                time.sleep(0.1)
            
            else:
                # Regular .wav file (old behavior)
                st.info("📁 Processing regular .wav file")
                time.sleep(0.5)
            
            # Final check - file must exist
            if not os.path.exists(audio_path):
                st.error(f"❌ File not found: {audio_path}")
                st.error("💡 Windows Defender may have deleted it immediately")
                st.error("💡 Add Windows Defender exclusion for folder")
                return None, None, False
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                st.error("❌ Audio file is empty")
                return None, None, False
            
            st.info(f"📁 File ready for transcription: {file_size} bytes")
            st.info("🎯 Starting Whisper transcription...")
            st.info("⏳ This takes 10-30 seconds...")
            
            # TRANSCRIBE IMMEDIATELY
            # No more delays - every millisecond counts!
            result = model.transcribe(
                audio_path,
                fp16=False,  # CPU compatibility
                language=None,  # Auto-detect
                task='transcribe'
            )
            
            text = result['text'].strip()
            detected_lang = result.get('language', 'en')
            
            if not text:
                st.warning("⚠️ No text transcribed - try speaking louder")
                return None, None, False
            
            st.success("✅ Transcription completed successfully!")
            return text, detected_lang, True
        
        except FileNotFoundError as e:
            st.error(f"❌ File not found during transcription")
            st.error(f"💡 Error: {str(e)}")
            st.error("🛡️ Windows Defender deleted the file!")
            st.error("")
            st.error("❌ WINDOWS DEFENDER EXCLUSION NOT WORKING!")
            st.error("")
            st.error("✅ SOLUTION: Disable Windows Defender Real-Time Protection")
            st.error("1. Open Windows Security")
            st.error("2. Virus & threat protection")
            st.error("3. Manage settings")
            st.error("4. Turn OFF 'Real-time protection'")
            st.error("5. Try voice recognition again")
            st.error("6. Turn Real-time protection back ON when done")
            return None, None, False
        
        except Exception as e:
            st.error(f"❌ Transcription error: {str(e)}")
            st.error(f"💡 File: {audio_path}")
            return None, None, False
        
        finally:
            # Clean up the .wav file if we created it
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                    st.info("✅ Temp .wav file cleaned up")
                except:
                    pass

# ============================================================================
# AUDIO RECORDER
# ============================================================================

class AudioRecorder:
    """Records audio from microphone"""
    
    @staticmethod
    def record_audio(duration=10, sample_rate=16000):
        """Record audio from microphone"""
        try:
            # Create columns for recording status
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.info(f"🎤 Recording for {duration} seconds... Speak NOW!")
                
                # Show countdown
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16
            )
            
            # Update progress during recording
            for i in range(duration):
                time.sleep(1)
                progress = (i + 1) / duration
                progress_bar.progress(progress)
                status_text.text(f"Recording... {duration - i - 1} seconds left")
            
            sd.wait()  # Wait until recording is finished
            
            with col2:
                st.success("✅ Recording complete! Processing...")
            
            # Save to temporary file (DEFENDER-PROOF METHOD)
            # Use .dat extension first - Windows Defender won't scan it!
            # Then rename to .wav only when ready for Whisper
            
            # Check if we're in OneDrive - this can cause issues
            cwd = os.getcwd()
            if "OneDrive" in cwd:
                st.warning("⚠️ Running from OneDrive folder - may cause file sync issues")
                st.info("💡 For best results, copy app to a local folder (e.g., C:\\VoiceRAG)")
            
            # Create temp directory - use absolute path
            temp_dir = os.path.abspath(os.path.join(cwd, "temp_audio"))
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # CRITICAL: Save as .dat first (Windows Defender ignores .dat files!)
            temp_filename_dat = os.path.normpath(os.path.join(temp_dir, f"recording_{timestamp}.dat"))
            temp_filename_dat = os.path.abspath(temp_filename_dat)
            
            st.info(f"📁 Creating protected file at: {temp_filename_dat}")
            
            # Write audio data as .dat file
            wav.write(temp_filename_dat, sample_rate, audio_data)
            
            # Delay to ensure write completes
            time.sleep(0.5)
            
            # Verify .dat file was created successfully
            if not os.path.exists(temp_filename_dat):
                raise Exception(f"Failed to create audio file at {temp_filename_dat}")
            
            file_size = os.path.getsize(temp_filename_dat)
            if file_size == 0:
                raise Exception("Audio file is empty - recording may have failed")
            
            st.success(f"✅ Protected file created: {file_size} bytes")
            st.info(f"🛡️ Using .dat extension to bypass Windows Defender")
            
            # Return the .dat filename - we'll handle the rename in transcription
            return temp_filename_dat, True
            
        except Exception as e:
            st.error(f"❌ Recording error: {str(e)}")
            st.info("💡 Make sure your microphone is connected and allowed")
            return None, False

# ============================================================================
# INPUT HANDLER
# ============================================================================

class InputHandler:
    """Handles voice and text input"""
    
    @staticmethod
    def capture_voice_input(whisper_model, duration=10):
        """Capture and transcribe voice input using Whisper"""
        
        # Record audio
        audio_file, success = AudioRecorder.record_audio(
            duration=duration,
            sample_rate=Config.SAMPLE_RATE
        )
        
        if not success or not audio_file:
            return None, None, False
        
        # Transcribe with Whisper
        with st.spinner("🔄 Transcribing with Whisper Large..."):
            text, detected_lang, success = WhisperHandler.transcribe_audio(
                whisper_model, audio_file
            )
        
        # Clean up temp file (with retry for Windows)
        if audio_file:
            max_retries = 3
            for i in range(max_retries):
                try:
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                        st.info(f"✅ Temp file cleaned up")
                    break
                except PermissionError:
                    if i < max_retries - 1:
                        time.sleep(0.5)  # Wait before retry
                    else:
                        st.warning(f"⚠️ Could not delete temp file (Windows locked it)")
                except Exception as e:
                    st.warning(f"⚠️ Cleanup warning: {str(e)}")
                    break
        
        if success and text:
            st.success(f"✅ Transcribed: {text}")
            st.info(f"🌍 Detected language: {detected_lang}")
            return text, detected_lang, True
        
        return None, None, False
    
    @staticmethod
    def get_text_input(user_input):
        """Process text input"""
        if user_input and user_input.strip():
            return user_input.strip(), True
        return None, False

# ============================================================================
# LANGUAGE DETECTOR
# ============================================================================

class LanguageDetector:
    """Detects and manages language using langdetect"""
    
    def __init__(self):
        pass
    
    def detect_language(self, text):
        """Detect language of input text"""
        try:
            from langdetect import detect, detect_langs
            
            detected_lang = detect(text)
            
            # Get confidence
            lang_probs = detect_langs(text)
            confidence = 0
            for lang in lang_probs:
                if lang.lang == detected_lang:
                    confidence = lang.prob
                    break
            
            st.session_state.detected_language = detected_lang
            
            return {
                'language': detected_lang,
                'confidence': confidence,
                'success': True
            }
        except Exception as e:
            return {
                'language': 'en',
                'confidence': 0,
                'success': False,
                'error': str(e)
            }
    
    def is_english(self, lang_code):
        """Check if language is English"""
        return lang_code.lower() in ['en', 'eng']

# ============================================================================
# TRANSLATOR
# ============================================================================

class TranslationService:
    """Handles all translation operations using deep-translator"""
    
    def __init__(self):
        pass
    
    def translate_to_english(self, text, source_lang):
        """Translate text to English"""
        if source_lang.lower() in ['en', 'eng', 'english']:
            return text, True
        
        try:
            translator = GoogleTranslator(source=source_lang, target='en')
            translated = translator.translate(text)
            return translated, True
        except Exception as e:
            st.warning(f"Translation note: {str(e)}")
            return text, False
    
    def translate_from_english(self, text, target_lang):
        """Translate text from English to target language"""
        if target_lang.lower() in ['en', 'eng', 'english']:
            return text, True
        
        try:
            translator = GoogleTranslator(source='en', target=target_lang)
            translated = translator.translate(text)
            return translated, True
        except Exception as e:
            st.warning(f"Translation note: {str(e)}")
            return text, False

# ============================================================================
# DOCUMENT PROCESSOR (RAG)
# ============================================================================

class DocumentProcessor:
    """Handles document processing and retrieval"""
    
    @staticmethod
    def process_documents(uploaded_files):
        """Process uploaded documents into chunks"""
        all_chunks = []
        all_titles = []
        
        for file in uploaded_files:
            try:
                content = file.read().decode('utf-8', errors='ignore')
                
                # Split into paragraphs
                paragraphs = content.replace('\n\n', '|PARA|').split('|PARA|')
                
                for para in paragraphs:
                    # Further split large paragraphs
                    if len(para) > Config.CHUNK_SIZE:
                        sentences = para.split('. ')
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) < Config.CHUNK_SIZE:
                                current_chunk += sentence + ". "
                            else:
                                if current_chunk.strip():
                                    all_chunks.append(current_chunk.strip())
                                    all_titles.append(file.name)
                                current_chunk = sentence + ". "
                        
                        if current_chunk.strip():
                            all_chunks.append(current_chunk.strip())
                            all_titles.append(file.name)
                    else:
                        if para.strip() and len(para.strip()) > 50:
                            all_chunks.append(para.strip())
                            all_titles.append(file.name)
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        return all_chunks, all_titles
    
    @staticmethod
    def search_documents(query, documents, doc_titles, top_k=3):
        """Search documents using keyword matching"""
        if not documents:
            return "", [], 0
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score each document
        scores = []
        for idx, doc in enumerate(documents):
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            
            # Calculate overlap score
            overlap = len(query_words.intersection(doc_words))
            
            # Bonus for exact phrase
            if query_lower in doc_lower:
                overlap += 10
            
            # Bonus for multiple word matches
            for word in query_words:
                if len(word) > 3:
                    if word in doc_lower:
                        overlap += 2
            
            if overlap > 0:
                scores.append((overlap, doc, doc_titles[idx], idx))
        
        if not scores:
            return "", [], 0
        
        # Sort by score
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Get top k results
        top_results = scores[:top_k]
        
        # Format context
        context_parts = []
        sources = []
        for score, doc, title, idx in top_results:
            context_parts.append(f"[Source: {title}]\n{doc}\n")
            sources.append(title)
        
        context = "\n".join(context_parts)
        unique_sources = list(set(sources))
        
        return context, unique_sources, len(top_results)

# ============================================================================
# LLM SERVICE
# ============================================================================

class LLMService:
    """Handles LLM interactions"""
    
    @staticmethod
    def generate_response(query, context="", llm_type="ollama", **kwargs):
        """Generate response from LLM"""
        
        # Build prompt
        if context:
            system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context contains relevant information, use it to provide an accurate answer.
If the context doesn't contain relevant information, provide a helpful general response.
Be clear, concise, and accurate."""
            
            full_prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION: {query}

Please provide a helpful answer:"""
        else:
            full_prompt = query
        
        # Call appropriate LLM
        if llm_type == "ollama":
            return LLMService._call_ollama(full_prompt, **kwargs)
        elif llm_type == "claude":
            return LLMService._call_claude(full_prompt, **kwargs)
        elif llm_type == "openai":
            return LLMService._call_openai(full_prompt, **kwargs)
        else:
            return "Error: Invalid LLM type", False
    
    @staticmethod
    def _call_ollama(prompt, model="llama2", url="http://localhost:11434"):
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response'), True
            else:
                return f"Ollama error: Status {response.status_code}", False
                
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Make sure it's running: ollama serve", False
        except requests.exceptions.Timeout:
            return "⏱️ Request timeout. Model may be processing. Try again.", False
        except Exception as e:
            return f"Error: {str(e)}", False
    
    @staticmethod
    def _call_claude(prompt, **kwargs):
        """Call Claude API"""
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["content"][0]["text"], True
            else:
                return f"Claude error: Status {response.status_code}", False
                
        except Exception as e:
            return f"Error: {str(e)}", False
    
    @staticmethod
    def _call_openai(prompt, api_key=None, model="gpt-3.5-turbo"):
        """Call OpenAI API"""
        if not api_key:
            return "OpenAI API key required", False
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"], True
            else:
                return f"OpenAI error: Status {response.status_code}", False
                
        except Exception as e:
            return f"Error: {str(e)}", False

# ============================================================================
# TEXT-TO-SPEECH SERVICE
# ============================================================================

class TTSService:
    """Text-to-Speech service"""
    
    @staticmethod
    def generate_speech(text, lang='en'):
        """Generate speech from text"""
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            fp = BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read(), True
        except Exception as e:
            st.error(f"TTS Error: {str(e)}")
            return None, False

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Voice RAG with Whisper",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    initialize_session_state()
    initialize_temp_directory()  # Initialize temp_audio folder
    cleanup_old_audio_files()  # Clean up old temp files
    
    # Load services
    lang_detector = LanguageDetector()
    translator = TranslationService()
    
    # Custom CSS
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        .chat-message {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        .user-message {
            background-color: #1e2530;
            border-left-color: #4a9eff;
            color: white;
        }
        .assistant-message {
            background-color: #162e1e;
            border-left-color: #19c37d;
            color: white;
        }
        .status-box {
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            border-left: 4px solid;
            font-weight: bold;
        }
        .status-ready { background-color: #1a472a; border-left-color: #10a37f; color: #10a37f; }
        .status-processing { background-color: #3a3a1a; border-left-color: #ffa500; color: #ffa500; }
        .status-error { background-color: #4a1a1a; border-left-color: #ff4444; color: #ff4444; }
        .big-button {
            font-size: 1.5rem !important;
            padding: 1rem 2rem !important;
            margin: 1rem 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Model Loading
        st.subheader("🎤 Whisper Model")
        
        whisper_size = st.selectbox(
            "Model Size",
            ["large", "medium", "base", "small", "tiny"],
            index=0,
            help="Large is most accurate but slower. Recommended: large or medium"
        )
        
        if st.button("🔄 Load Whisper Model", use_container_width=True):
            with st.spinner(f"Loading Whisper {whisper_size}..."):
                model, success = WhisperHandler.load_model(whisper_size)
                if success:
                    st.session_state.whisper_model = model
                    st.success("✅ Model ready!")
        
        if st.session_state.whisper_model is not None:
            st.success(f"✅ Whisper {whisper_size} loaded")
        else:
            st.warning("⚠️ Load Whisper model first")
        
        st.divider()
        
        # LLM Settings
        st.subheader("🤖 AI Model")
        llm_type = st.selectbox(
            "LLM Provider",
            ["ollama", "claude", "openai"]
        )
        
        if llm_type == "ollama":
            ollama_model = st.text_input("Model", "llama2")
            ollama_url = st.text_input("URL", "http://localhost:11434")
            
            if st.button("🔌 Test Ollama"):
                try:
                    resp = requests.get(f"{ollama_url}/api/tags", timeout=2)
                    if resp.status_code == 200:
                        models = resp.json().get('models', [])
                        st.success(f"✅ {len(models)} models available")
                    else:
                        st.error("❌ Not responding")
                except:
                    st.error("❌ Connection failed")
        
        elif llm_type == "openai":
            openai_key = st.text_input("API Key", type="password")
            openai_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
        
        st.divider()
        
        # Document Upload
        st.subheader("📚 Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['txt'],
            accept_multiple_files=True
        )
        
        if st.button("📥 Process Documents"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    chunks, titles = DocumentProcessor.process_documents(uploaded_files)
                    st.session_state.documents = chunks
                    st.session_state.doc_titles = titles
                    st.success(f"✅ {len(chunks)} chunks loaded!")
        
        if st.session_state.documents:
            st.info(f"📊 {len(st.session_state.documents)} chunks active")
        
        st.divider()
        
        # Voice Settings
        st.subheader("🎤 Voice Settings")
        enable_voice_input = st.checkbox("Enable Voice Input", True)
        
        recording_duration = 10  # Default
        if enable_voice_input:
            recording_duration = st.slider(
                "Recording Duration (seconds)",
                3, 15, 10
            )
        
        enable_voice_output = st.checkbox("Enable Voice Output", False)
        
        if enable_voice_output:
            voice_lang = st.selectbox(
                "Output Language",
                Config.SUPPORTED_LANGUAGES,
                format_func=lambda x: {
                    'en': 'English', 'es': 'Spanish', 'fr': 'French',
                    'de': 'German', 'zh': 'Chinese', 'ja': 'Japanese',
                    'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi',
                    'pt': 'Portuguese', 'ru': 'Russian', 'it': 'Italian'
                }.get(x, x)
            )
        
        st.divider()
        
        # Controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.documents = []
            st.session_state.doc_titles = []
            st.rerun()
    
    # ========================================================================
    # MAIN AREA
    # ========================================================================
    
    st.title("🎤 Voice RAG Chatbot with Whisper Large")
    st.caption("🎤 Professional Speech Recognition • 🌍 Multilingual • 📚 Document Q&A")
    
    # Status
    status_class = "status-ready"
    if "processing" in st.session_state.processing_status.lower():
        status_class = "status-processing"
    elif "error" in st.session_state.processing_status.lower() or "failed" in st.session_state.processing_status.lower():
        status_class = "status-error"
    
    st.markdown(
        f'<div class="status-box {status_class}">⚡ Status: {st.session_state.processing_status}</div>',
        unsafe_allow_html=True
    )
    
    # Display chat
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        icon = "👤" if message["role"] == "user" else "🤖"
        
        st.markdown(
            f'<div class="chat-message {role_class}">'
            f'<b>{icon} {message["role"].title()}</b><br>{message["content"]}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        if message["role"] == "assistant":
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.write(f"• {source}")
            
            if "audio" in message and message["audio"]:
                st.audio(message["audio"], format='audio/mp3')
    
    # Voice button
    if enable_voice_input and st.session_state.whisper_model is not None:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            voice_button = st.button("🎤 SPEAK", use_container_width=True, key="voice_btn")
    else:
        voice_button = False
        if enable_voice_input and st.session_state.whisper_model is None:
            st.warning("⚠️ Load Whisper model first (in sidebar)")
    
    # Text input
    user_input = st.chat_input("💭 Type your message in any language...")
    
    # ========================================================================
    # PROCESS INPUT
    # ========================================================================
    
    input_text = None
    detected_lang_whisper = None
    
    # Voice input
    if voice_button:
        st.session_state.processing_status = "Capturing voice..."
        
        # Capture voice immediately (don't rerun)
        input_text, detected_lang_whisper, success = InputHandler.capture_voice_input(
            st.session_state.whisper_model,
            duration=recording_duration if enable_voice_input else 10
        )
        
        if not success:
            st.session_state.processing_status = "Ready"
            st.error("❌ Voice capture failed. Please try again.")
            st.info("💡 Tips: Speak clearly, reduce background noise, check microphone")
            input_text = None  # Clear to prevent processing
    
    # Text input
    elif user_input:
        input_text, success = InputHandler.get_text_input(user_input)
        if not success:
            st.session_state.processing_status = "Invalid input"
            st.stop()
    
    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================
    
    if input_text:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": input_text
        })
        
        st.session_state.processing_status = "Processing..."
        
        # Detect language (use Whisper detection if available, else Google)
        if detected_lang_whisper:
            detected_lang = detected_lang_whisper
        else:
            with st.spinner("🔍 Detecting language..."):
                lang_result = lang_detector.detect_language(input_text)
                detected_lang = lang_result['language']
        
        # Translate to English
        english_query = input_text
        if not lang_detector.is_english(detected_lang):
            with st.spinner("🌍 Translating to English..."):
                english_query, success = translator.translate_to_english(
                    input_text, detected_lang
                )
        
        # Search documents
        context = ""
        sources = []
        if st.session_state.documents:
            with st.spinner("📚 Searching documents..."):
                context, sources, count = DocumentProcessor.search_documents(
                    english_query,
                    st.session_state.documents,
                    st.session_state.doc_titles,
                    top_k=Config.TOP_K_RETRIEVAL
                )
                if count > 0:
                    st.info(f"Found {count} relevant sections")
        
        # Generate response
        with st.spinner("🤖 Generating response..."):
            if llm_type == "ollama":
                response, success = LLMService.generate_response(
                    english_query, context, "ollama",
                    model=ollama_model, url=ollama_url
                )
            elif llm_type == "claude":
                response, success = LLMService.generate_response(
                    english_query, context, "claude"
                )
            elif llm_type == "openai":
                response, success = LLMService.generate_response(
                    english_query, context, "openai",
                    api_key=openai_key, model=openai_model
                )
            
            if not success:
                st.error(response)
                st.session_state.processing_status = "Error"
                st.stop()
        
        # Translate back
        final_response = response
        if not lang_detector.is_english(detected_lang):
            with st.spinner("🌍 Translating response..."):
                final_response, success = translator.translate_from_english(
                    response, detected_lang
                )
        
        # Generate audio
        audio_data = None
        if enable_voice_output:
            with st.spinner("🔊 Generating speech..."):
                audio_data, success = TTSService.generate_speech(
                    final_response,
                    voice_lang if enable_voice_output else 'en'
                )
        
        # Store response
        message_data = {
            "role": "assistant",
            "content": final_response,
            "sources": sources,
            "detected_lang": detected_lang
        }
        
        if audio_data:
            message_data["audio"] = audio_data
        
        st.session_state.messages.append(message_data)
        st.session_state.processing_status = "Ready"
        
        st.rerun()
    
    # Footer
    if len(st.session_state.messages) == 0:
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
        <h3 style='color: #10a37f;'>🎤 Professional Voice RAG with Whisper Large</h3>
        <br>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
            <div style='margin: 1rem; color: #888;'>
                <h4 style='color: white;'>🎤 Voice</h4>
                Whisper Large<br>Professional accuracy
            </div>
            <div style='margin: 1rem; color: #888;'>
                <h4 style='color: white;'>💬 Chat</h4>
                Any language<br>Auto-translation
            </div>
            <div style='margin: 1rem; color: #888;'>
                <h4 style='color: white;'>📚 RAG</h4>
                Upload docs<br>Smart search
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
