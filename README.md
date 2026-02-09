\# 🧮 Math Mentor AI - JEE Level Math Assistant



\[!\[Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=Streamlit\&logoColor=white)](https://streamlit.io)

\[!\[Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge\&logo=python\&logoColor=white)](https://python.org)

\[!\[License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)



\*\*Bhai, ye ek advanced Math Mentor AI hai jo JEE-style math problems solve karta hai!\*\*



\## ✨ Features



\### 🎯 Core Capabilities

\- \*\*Multimodal Input\*\*: Image (OCR), Audio (Speech-to-Text), Text

\- \*\*RAG Pipeline\*\*: Knowledge retrieval from curated math database

\- \*\*Multi-Agent System\*\*: 5 specialized agents working together

\- \*\*Human-in-the-Loop\*\*: Human review for low-confidence solutions

\- \*\*Self-Learning Memory\*\*: Learns from corrections and feedback

\- \*\*Step-by-Step Explanations\*\*: Student-friendly explanations



\### 🤖 AI Agents

1\. \*\*Parser Agent\*\* - Input ko structured format mein convert karta hai

2\. \*\*Intent Router\*\* - Problem type identify karta hai

3\. \*\*Solver Agent\*\* - RAG + tools use karke solve karta hai

4\. \*\*Verifier Agent\*\* - Solution check karta hai

5\. \*\*Explainer Agent\*\* - Step-by-step explanation deta hai



\## 🚀 Quick Start



\### Prerequisites

\- Python 3.8 or higher

\- Tesseract OCR (for image processing)

\- FFmpeg (for audio processing)



\### Installation



```bash

\# 1. Clone repository

git clone https://github.com/yourusername/math-mentor-ai.git

cd math-mentor-ai



\# 2. Install dependencies

pip install -r requirements.txt



\# 3. Install system dependencies

\# Ubuntu/Debian:

sudo apt-get install tesseract-ocr ffmpeg



\# macOS:

brew install tesseract ffmpeg



\# Windows: Download Tesseract and FFmpeg manually



\# 4. Initialize knowledge base

python -c "from rag.vector\_store import initialize\_knowledge\_base; initialize\_knowledge\_base()"



\# 5. Run the app

streamlit run app.py

