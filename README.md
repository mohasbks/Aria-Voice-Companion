# Aria Voice Companion 🎙️✨

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue.svg?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Groq-API-f55036?style=for-the-badge" alt="Groq API" />
  <img src="https://img.shields.io/badge/Llama_3-8A2BE2?style=for-the-badge" alt="Llama 3" />
  <img src="https://img.shields.io/badge/JS_Vanilla-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="Vanilla JS" />
</div>

<br>

**Aria** is an ultra-fast, multi-language conversational AI voice companion. Designed to feel human, Aria leverages Groq for near-instant inference with Llama-3, WebSockets for fluid real-time communication, and a hybrid Text-to-Speech (TTS) engine powered by **Orpheus** (via Groq) and **Edge-TTS**.

Featuring a premium, minimalist neo-brutalist UI, Aria understands human emotions and adapts her voice and interface colors in real-time based on the flow of the conversation.

---

## 🚀 Features

- **Blazing Fast AI:** Utilizes Groq's LPU inference engine to achieve sub-second response times using `llama3-8b-8192`.
- **Hybrid TTS Engine:** High-fidelity human voices via CanopyLabs' Orpheus models (`orpheus-v1-english` & `orpheus-arabic-saudi`), with an automatic fallback to Edge-TTS.
- **Dynamic Emotion Tracking:** Aria infers the emotional context of the conversation and visually reflects it by changing the UI theme (colors, badges, waveforms) and her tone of voice.
- **Bilingual (English & Arabic):** Full support for switching personas and languages on the fly via the settings modal. 
- **Session Memory:** Stores complete conversation history locally in SQLite, retaining context per user session.
- **Native Telegram Bot:** Includes a fully functional Telegram bot (`telegram_bot.py`) that mirrors Aria's core brain and replies via high-quality voice notes.
- **Secure & Local API Keys:** Bring-Your-Own-Key (BYOK). Users can inject their own Groq API keys locally inside the browser without exposing them to the backend server.

---

## 🛠 Tech Stack

### Backend
- **Python 3.13** + **FastAPI**
- **WebSockets** (Real-time duplex streaming)
- **SQLite** (Session/Memory management)
- **python-telegram-bot** (Telegram Integration)

### Frontend
- **Vanilla JavaScript** (Zero bloated frameworks)
- **Web Speech API & MediaRecorder** (Client-side STT options)
- **CSS3 Variables & Animations** (Dynamic Emotion Theming)

---

## ⚙️ Installation & Usage

### 1. Requirements
- Python 3.10+ (Tested on 3.13)
- A [Groq API Key](https://console.groq.com/)
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather) (Optional)

### 2. Setup
Clone the repository and install the dependencies:

```bash
git clone https://github.com/mohasbks/Aria-Voice-Companion.git
cd Aria-Voice-Companion/backend
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file inside the root directory and populate it:

```env
GROQ_API_KEY=gsk_your_default_english_key_here
GROQ_API_KEY_ARABIC=gsk_your_arabic_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### 4. Running the Web App
Start the FastAPI Uvicorn server:
```bash
cd backend
python main.py
```
Open your browser and navigate to `http://localhost:8000`

### 5. Running the Telegram Bot
To run the companion bot on Telegram in parallel:
```bash
cd backend
python telegram_bot.py
```

---

## 🎨 UI Showcase

The web interface is built using a dark, sleek design that reacts to Aria's mood. If you insult her, the screen turns cold and indigo; if you joke with her, the screen warms up to vibrant pinks and golds.

*(Wait for the audio wave animations while she generates the voice response!)*

---

## 📝 License
This project is open-source and available under the MIT License. Feel free to fork, modify, and integrate into your own SaaS tools!

---
*Built with ❤️ for real-time human-AI interaction.*
