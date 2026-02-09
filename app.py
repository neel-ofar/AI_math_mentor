"""
Math Mentor AI - Complete Application
JEE Level Math Assistant with Multi-Agent System
"""

import streamlit as st
import tempfile
import os
from PIL import Image
import numpy as np
import pytesseract
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Windows ke liye tesseract path set karo
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    pass  # Linux/Mac pe default use hoga

# Import hamare custom modules
from utils.ocr_handler import extract_text_from_image
from utils.audio_handler import transcribe_audio, transcribe_recording
from agents.parser_agent import ParserAgent
from rag.retriever import retrieve_context
from agents.solver_agent import SolverAgent
from agents.verifier_agent import VerifierAgent
from agents.explainer_agent import ExplainerAgent
from memory.simple_memory_handler import MemoryHandler

# Page config
st.set_page_config(
    page_title="Math Mentor AI",
    page_icon="🧮",
    layout="wide"
)

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory = MemoryHandler()

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'input_method' not in st.session_state:
    st.session_state.input_method = "text"

if 'show_memory' not in st.session_state:
    st.session_state.show_memory = False

if 'confidence_threshold' not in st.session_state:
    st.session_state.confidence_threshold = 0.70

# OCR specific session states
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'ocr_confidence' not in st.session_state:
    st.session_state.ocr_confidence = 0.0
if 'text_extracted' not in st.session_state:
    st.session_state.text_extracted = False
if 'question_text' not in st.session_state:
    st.session_state.question_text = ""

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E3A8A;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .input-box {
        background-color: #F3F4F6;
        padding: 25px;
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        margin: 20px 0;
    }
    .solution-box {
        background-color: #ECFDF5;
        padding: 25px;
        border-radius: 10px;
        border: 2px solid #10B981;
        margin: 20px 0;
    }
    .stats-box {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #3B82F6;
        margin: 10px 0;
    }
    .agent-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .ocr-success {
        background-color: #d1fae5;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #10b981;
        margin: 10px 0;
    }
    .ocr-warning {
        background-color: #fef3c7;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #f59e0b;
        margin: 10px 0;
    }
    .audio-tab {
        background: #f8fafc;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='main-header'><h1>🧮 Math Mentor AI</h1></div>", unsafe_allow_html=True)
    
    # Input Method
    st.markdown("### Input Method")
    input_method = st.radio(
        "Kaise input dena chahte ho?",
        ["Text", "Image", "Audio"],
        key="input_method_radio",
        label_visibility="collapsed"
    )
    st.session_state.input_method = input_method.lower()
    
    # Confidence Threshold
    st.markdown("---")
    st.markdown("### Confidence Level")
    confidence = st.slider(
        "Solution accuracy level",
        0.0, 1.0, 0.70, 0.05,
        label_visibility="collapsed"
    )
    st.session_state.confidence_threshold = confidence
    
    # Memory Bank Toggle
    st.markdown("---")
    show_memory = st.toggle("Show Memory Bank", value=False)
    st.session_state.show_memory = show_memory
    
    # Memory Stats
    st.markdown("---")
    st.markdown("<div class='stats-box'>", unsafe_allow_html=True)
    st.markdown("### 📊 Performance")
    stats = st.session_state.memory.get_stats()
    total = stats.get('total', 0)
    correct = stats.get('correct', 0)
    
    st.metric("Total Problems", total)
    st.metric("Correct Solutions", correct)
    
    if total > 0:
        accuracy = (correct / total) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    else:
        st.metric("Accuracy", "0%")
    
    # AI Status
    st.markdown("---")
    st.markdown("### 🤖 AI Status")
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success("✓ AI Active")
        st.caption("Using Groq LLM")
    else:
        st.error("✗ AI Offline")
        st.caption("Add GROQ_API_KEY to .env file")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main Content
st.markdown("<div class='main-header'><h2>Math Mentor AI - JEE Level Assistant</h2></div>", unsafe_allow_html=True)

st.markdown("""
**Bhai, yahan par aap:**

1. 📸 **Photo khich ke upload karo**  
2. 🎤 **Audio record/upload karo**  
3. ⌨️ **Direct type karo**

AI aapko step-by-step solution dega with JEE level explanations!
""")

st.markdown("---")

# Multi-Agent System Info
with st.expander("🤖 **Multi-Agent System Architecture**", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("**🔍 Parser Agent**")
        st.caption("Question ko analyze karta hai")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("**🧠 Solver Agent**")
        st.caption("Problem solve karta hai")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("**✅ Verifier Agent**")
        st.caption("Solution check karta hai")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("**📖 Explainer Agent**")
        st.caption("Step-by-step batata hai")
        st.markdown("</div>", unsafe_allow_html=True)

# Question Input Section
st.markdown("<div class='input-box'>", unsafe_allow_html=True)
st.markdown("### 📝 Question Input")
st.markdown("**Math question likhe yahan:**")

# Reset question_text at start of each run
question_text = ""
input_received = False

if st.session_state.input_method == "text":
    user_input = st.text_area(
        "Math question likho yahan:",
        height=150,
        placeholder="Example: Find the derivative of x^2 + 3x - 5\nOr: What is the probability of getting heads in a coin toss?\nOr: Solve 2x + 5 = 13",
        key="text_input"
    )
    if user_input and user_input.strip():
        question_text = user_input.strip()
        input_received = True
        st.session_state.question_text = question_text

elif st.session_state.input_method == "image":
    uploaded_image = st.file_uploader(
        "Math problem ki photo upload karo", 
        type=['jpg', 'jpeg', 'png'],
        key="image_uploader"
    )
    
    if uploaded_image is not None:
        # Display image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", width='stretch')
        
        # Two columns layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔍 Extract Text from Image", width='stretch', key="extract_btn"):
                with st.spinner("Reading text from image..."):
                    # Extract text
                    extracted_text, confidence = extract_text_from_image(image)
                    
                    # Store in session state
                    st.session_state.extracted_text = extracted_text
                    st.session_state.ocr_confidence = confidence
                    st.session_state.text_extracted = True
                    st.session_state.question_text = extracted_text
                    
                    st.success(f"✅ Text extracted with {confidence:.1%} confidence!")
                    st.rerun()
        
        with col2:
            st.info("**Tips:**\n• Clear, well-lit images\n• Avoid glare/shadows\n• Typed text > Handwritten")
        
        # Show extracted text if available
        if st.session_state.text_extracted:
            st.markdown("---")
            st.subheader("📝 Extracted Text")
            
            # Show confidence with appropriate styling
            confidence = st.session_state.ocr_confidence
            if confidence > 0.7:
                st.markdown(f'<div class="ocr-success">✅ High confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
            elif confidence > 0.4:
                st.markdown(f'<div class="ocr-warning">⚠️ Medium confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
            else:
                st.error(f"❌ Low confidence: {confidence:.1%}")
            
            # Editable text area
            current_text = st.session_state.extracted_text
            edited_text = st.text_area(
                "Edit the extracted text if needed:",
                value=current_text,
                height=150,
                key="ocr_editable_text"
            )
            
            # Update session state with edited text
            st.session_state.question_text = edited_text
            
            # Action buttons
            col_accept, col_redo = st.columns(2)
            
            with col_accept:
                if st.button("✅ Use This Text", width='stretch', type="primary", key="use_text"):
                    question_text = edited_text
                    input_received = True
                    st.session_state.question_text = edited_text
                    st.success("✅ Text ready for solving!")
            
            with col_redo:
                if st.button("🔄 Re-extract Text", width='stretch', key="re_extract"):
                    # Reset OCR state
                    st.session_state.text_extracted = False
                    st.session_state.extracted_text = ""
                    st.session_state.ocr_confidence = 0.0
                    st.rerun()
            
            # Auto-set for solving if text is valid
            if edited_text and len(edited_text.strip()) > 5:
                question_text = edited_text
                input_received = True
                st.session_state.question_text = edited_text

elif st.session_state.input_method == "audio":
    # Create tabs for upload and record
    tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Record Live Audio"])
    
    with tab1:
        st.markdown('<div class="audio-tab">', unsafe_allow_html=True)
        
        audio_file = st.file_uploader(
            "Audio file upload karo (MP3, WAV, M4A, OGG)",
            type=['mp3', 'wav', 'm4a', 'ogg', 'webm'],
            key="audio_uploader"
        )
        
        if audio_file is not None:
            # Show audio player
            st.audio(audio_file, format="audio/wav")
            
            if st.button("🎤 Transcribe Uploaded Audio", width='stretch', key="transcribe_upload"):
                with st.spinner("Converting audio to text..."):
                    # Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(audio_file.read())
                        audio_path = tmp.name
                    
                    try:
                        # Use the new function
                        extracted_text, confidence = transcribe_audio(audio_path)
                        
                        st.subheader("🎧 Transcribed Text")
                        st.text_area("Transcription:", extracted_text, height=150, key="transcription", label_visibility="collapsed")
                        st.write(f"**Confidence:** {confidence:.2%}")
                        
                        # Store in session state
                        st.session_state.question_text = extracted_text
                        
                        # Check confidence
                        if confidence < st.session_state.confidence_threshold:
                            st.warning("⚠️ Transcription confidence kam hai. Confirm karo:")
                            
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("✅ Haan, sahi hai", key="audio_yes", width='stretch'):
                                    st.session_state.question_text = extracted_text
                                    st.success("✅ Text ready for solving!")
                                    st.rerun()
                            with col_no:
                                edited_text = st.text_area("Corrected transcription:", extracted_text, 
                                                         height=150, key="audio_corrected")
                                if st.button("Submit Correction", key="audio_submit", width='stretch'):
                                    st.session_state.question_text = edited_text
                                    st.success("✅ Text updated!")
                                    st.rerun()
                        else:
                            st.session_state.question_text = extracted_text
                            st.success("✅ Good quality transcription! Ready to solve.")
                            
                    except Exception as e:
                        st.error(f"❌ Transcription error: {str(e)}")
                        st.info("Try uploading a clearer audio file or use text input")
                    finally:
                        # Cleanup
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="audio-tab">', unsafe_allow_html=True)
        st.markdown("### 🎤 Live Audio Recording")
        
        # Check if streamlit-mic-recorder is available
        try:
            import streamlit_mic_recorder
            
            # Create recorder
            audio = streamlit_mic_recorder.mic_recorder(
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                just_once=False,
                use_container_width=False,
                format="wav",
                key="recorder"
            )
            
            if audio is not None and audio.get('bytes'):
                # Display the recorded audio
                st.audio(audio['bytes'], format="audio/wav")
                
                if st.button("📝 Transcribe Recording", width='stretch', key="transcribe_record"):
                    with st.spinner("Processing your recording..."):
                        try:
                            # Use the recording function
                            extracted_text, confidence = transcribe_recording(
                                audio['bytes'],
                                sample_rate=audio.get('sample_rate', 16000)
                            )
                            
                            st.subheader("🎤 Your Recording Text")
                            st.text_area("Transcription:", extracted_text, height=150, key="record_transcription", label_visibility="collapsed")
                            st.write(f"**Confidence:** {confidence:.2%}")
                            
                            # Store in session state
                            st.session_state.question_text = extracted_text
                            
                            if confidence < 0.5:
                                st.warning("⚠️ Low confidence. Please speak clearly and try again.")
                                st.info("Tips: Speak slowly, avoid background noise, use a good microphone")
                            else:
                                st.success("✅ Recording transcribed successfully! Ready to solve.")
                                
                        except Exception as e:
                            st.error(f"❌ Recording transcription failed: {str(e)}")
                            
        except ImportError:
            st.warning("⚠️ Live recording requires additional package")
            st.code("pip install streamlit-mic-recorder")
            st.info("For now, please use the upload option above")
            
        except Exception as e:
            st.error(f"❌ Recording error: {str(e)}")
            st.info("Please use audio upload or text input")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Help section
    with st.expander("💡 Audio Tips", expanded=False):
        st.markdown("""
        **For best results:**
        
        🎤 **Recording Tips:**
        - Speak clearly and at normal pace
        - Use a good microphone
        - Avoid background noise
        - Keep recordings under 30 seconds
        
        📁 **File Tips:**
        - WAV format works best
        - Clear audio without distortion
        - Mono recordings preferred
        - Sample rate: 16kHz or higher
        
        🔧 **Troubleshooting:**
        - Install: `pip install SpeechRecognition pydub`
        - Install ffmpeg for format conversion
        - Check microphone permissions
        """)

st.markdown("</div>", unsafe_allow_html=True)

# Solve Button - Get question_text from appropriate source
current_question_text = ""
if st.session_state.input_method == "text" and 'text_input' in st.session_state:
    current_question_text = st.session_state.text_input
elif st.session_state.input_method == "image" and st.session_state.get('question_text'):
    current_question_text = st.session_state.question_text
elif st.session_state.input_method == "audio" and st.session_state.get('question_text'):
    current_question_text = st.session_state.question_text

if st.button("🚀 Solve Math Problem", type="primary", width='stretch', key="solve_btn"):
    # Get the question text
    question_text = current_question_text
    
    if not question_text or len(question_text.strip()) < 5:
        st.warning("⚠️ Please enter or extract a question first!")
        st.info("For images: Click 'Extract Text' then 'Use This Text'")
        st.stop()
    
    # Check Groq API Key
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error("""
        **AI System Not Configured!**
        
        Please add your Groq API key to `.env` file:
        
        ```env
        GROQ_API_KEY=your_groq_api_key_here
        ```
        
        Get free API key from: https://console.groq.com/keys
        """)
        st.stop()
    
    # Create progress containers
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Step 1: Parse
        status_text.text("🔍 Step 1: Parsing question...")
        parser = ParserAgent()
        parsed_problem = parser.parse(question_text)
        progress_bar.progress(20)
        
        # Show parsed output
        with st.expander("📋 Parsed Problem Structure", expanded=False):
            st.json(parsed_problem)
        
        # Step 2: Check if needs clarification
        if parsed_problem.get('needs_clarification', False):
            st.warning("⚠️ Parser ko clarification chahiye!")
            clarification = st.text_input(
                "Kya ye question clear nahi hai? Please clarify:",
                value=parsed_problem.get('clarification_question', ''),
                key="clarification_input"
            )
            if st.button("Submit Clarification", key="clarify_submit", width='stretch'):
                if clarification:
                    parsed_problem['clarification'] = clarification
                    parsed_problem['needs_clarification'] = False
                    st.rerun()
            st.stop()
        
        # Step 3: Retrieve context from RAG
        status_text.text("📚 Step 2: Retrieving relevant knowledge...")
        context = retrieve_context(parsed_problem['problem_text'])
        progress_bar.progress(40)
        
        # Show retrieved context
        with st.expander("🧠 Retrieved Knowledge", expanded=False):
            if context:
                for i, doc in enumerate(context[:3]):  # Show top 3
                    with st.container():
                        st.write(f"**Source {i+1}** (Relevance: {doc.get('score', 0):.3f})")
                        st.write(doc['content'][:250] + "..." if len(doc['content']) > 250 else doc['content'])
                        st.divider()
            else:
                st.write("No relevant knowledge found in database")
        
        # Step 4: Solve using Groq
        status_text.text("🔄 Step 3: Solving problem...")
        solver = SolverAgent(use_llm=True)  # Always use LLM now
        solution = solver.solve(parsed_problem, context)
        progress_bar.progress(60)
        
        # Show solution steps
        with st.expander("🔧 Solution Steps", expanded=False):
            steps = solution.get('solution_steps', [])
            if steps:
                for i, step in enumerate(steps):
                    if isinstance(step, dict):
                        st.write(f"**Step {i+1}:** {step.get('expression', '')} → {step.get('result', '')}")
                    else:
                        st.write(f"**Step {i+1}:** {step}")
            else:
                st.write("No detailed steps available")
        
        # Step 5: Verify
        status_text.text("✅ Step 4: Verifying solution...")
        verifier = VerifierAgent()
        verification = verifier.verify(solution, parsed_problem)
        progress_bar.progress(80)
        
        # Step 6: Explain
        status_text.text("📖 Step 5: Creating explanation...")
        explainer = ExplainerAgent(use_llm=True)  # Always use LLM now
        explanation = explainer.explain(solution, verification)
        progress_bar.progress(100)
        status_text.text("🎉 Complete!")
        
        # Results Section
        st.success("✅ Problem solved successfully!")
        
        # Display results
        st.markdown("<div class='solution-box'>", unsafe_allow_html=True)
        
        # Answer
        st.markdown("### 🎯 Answer")
        answer = solution.get('final_answer', 'No answer found')
        if isinstance(answer, dict):
            if 'steps' in answer:
                st.write("**Step-by-step:**")
                for step in answer.get('steps', []):
                    st.write(f"• {step}")
            elif 'method' in answer:
                st.write(f"**Method:** {answer.get('method')}")
                if 'note' in answer:
                    st.write(f"**Note:** {answer.get('note')}")
        else:
            st.code(str(answer), language='python')
        
        # Confidence
        st.markdown("### 📊 Confidence")
        confidence_score = verification.get('confidence', 0.5)
        if confidence_score > 0.8:
            st.success(f"High: {confidence_score:.2%}")
        elif confidence_score > 0.5:
            st.warning(f"Medium: {confidence_score:.2%}")
        else:
            st.error(f"Low: {confidence_score:.2%}")
        
        # Verification Status
        st.markdown("### ✅ Verification Status")
        if verification.get('is_correct', False):
            st.success("✅ Solution verified as correct")
        else:
            st.error("❌ Issues found in solution")
            issues = verification.get('issues', [])
            if issues:
                st.write("**Issues:**")
                for issue in issues:
                    st.write(f"• {issue}")
        
        # Explanation
        st.markdown("### 📚 Step-by-Step Explanation")
        exp_text = explanation.get('explanation', 'No explanation generated')
        if isinstance(exp_text, dict):
            exp_text = exp_text.get('explanation', str(exp_text))
        
        st.markdown(exp_text)
        
        st.caption(f"🤖 Powered by: {solution.get('method', 'Groq LLM')}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Memory storage
        problem_id = st.session_state.memory.store(
            original_input=question_text,
            parsed_problem=parsed_problem,
            solution=solution,
            verification=verification,
            explanation=explanation
        )
        
        if problem_id:
            st.session_state.memory.last_id = problem_id
        
        # Feedback section
        st.divider()
        st.markdown("### 💬 Feedback")
        
        col_fb1, col_fb2 = st.columns(2)
        
        with col_fb1:
            if st.button("✅ Correct Solution", width='stretch', key="fb_correct"):
                if hasattr(st.session_state.memory, 'last_id') and st.session_state.memory.last_id:
                    st.session_state.memory.add_feedback(
                        problem_id=st.session_state.memory.last_id,
                        is_correct=True,
                        feedback="User marked as correct"
                    )
                    st.success("Thanks for feedback!")
                    st.rerun()
        
        with col_fb2:
            if st.button("❌ Incorrect Solution", width='stretch', key="fb_incorrect"):
                feedback = st.text_input("Kya galat hai?", key="fb_input")
                if feedback and hasattr(st.session_state.memory, 'last_id') and st.session_state.memory.last_id:
                    st.session_state.memory.add_feedback(
                        problem_id=st.session_state.memory.last_id,
                        is_correct=False,
                        feedback=feedback
                    )
                    st.success("Thanks for correction!")
                    st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        import traceback
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

# Memory Bank
if st.session_state.show_memory:
    st.markdown("---")
    st.markdown("### 🗂️ Memory Bank")
    memories = st.session_state.memory.get_all(limit=10)
    
    if memories:
        for mem in memories:
            if isinstance(mem, dict):
                problem_text = mem.get('parsed_problem', {}).get('problem_text', 
                           mem.get('problem', 
                           mem.get('problem_text', 'Unknown problem')))
                
                with st.expander(f"📝 {str(problem_text)[:60]}..."):
                    col_m1, col_m2 = st.columns(2)
                    
                    with col_m1:
                        st.write(f"**Topic:** {mem.get('parsed_problem', {}).get('topic', 
                                  mem.get('topic', 'Unknown'))}")
                        st.write(f"**Solved:** {mem.get('timestamp', 'Unknown')}")
                    
                    with col_m2:
                        is_correct = "Not rated"
                        if 'verification' in mem and isinstance(mem['verification'], dict):
                            is_correct = "✅ Yes" if mem['verification'].get('is_correct') else "❌ No"
                        elif 'is_correct' in mem:
                            is_correct = "✅ Yes" if mem['is_correct'] else "❌ No"
                        
                        st.write(f"**Correct:** {is_correct}")
                        answer = mem.get('solution', {}).get('final_answer', 
                                 mem.get('answer', 'No answer'))
                        st.write(f"**Answer:** {str(answer)[:50]}...")
    else:
        st.info("No problems stored in memory yet")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ by Math Mentor AI | JEE Level Math Assistant</p>
    <p>🤖 Multi-Agent System | 🧠 RAG Pipeline | 👨‍🏫 Human-in-the-Loop | 💾 Self-learning Memory</p>
</div>
""", unsafe_allow_html=True)
