<div align="center">
  <br>
  <!-- Beautiful Audio Wave / Glowing Orb representation -->
  <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHhicWFjcWNvMndzbDFocDNxZm4xcHF3czMweHU1cnpva2Y3bTkybCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/27L5Yz2bP2w3rtFdxH/giphy.gif" alt="Aria Audio Wave" width="250" style="border-radius: 50%; box-shadow: 0 0 50px rgba(100, 200, 255, 0.4);"/>
  <br><br>

  # 🎙️ Aria: The Empathetic Voice Companion  
  *Ultra-fast. Emotionally aware. Multi-lingual.*

  <p align="center">
    <a href="#"><img src="https://img.shields.io/badge/Python-3.13-blue.svg?style=for-the-badge&logo=python&color=0c0c0e" alt="Python" /></a>
    <a href="#"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&color=0c0c0e" alt="FastAPI" /></a>
    <a href="#"><img src="https://img.shields.io/badge/Groq%20LPU-Ultra%20Fast-f55036?style=for-the-badge&color=0c0c0e" alt="Groq API" /></a>
    <a href="#"><img src="https://img.shields.io/badge/Llama_3-7B-8A2BE2?style=for-the-badge&color=0c0c0e" alt="Llama 3" /></a>
  </p>

  <sub>Built with ❤️ combining <b>Vanilla JS</b>, <b>Orpheus TTS</b>, and <b>WebSockets</b> for sub-second, real-time interactions.</sub>
</div>

---

## ✨ Why Aria?

**Aria is not just another chatbot.** She is designed to feel alive.  
Using **Groq's hyper-fast Llama-3**, she listens to you, reads the context, naturally anticipates your mood, and changes her **voice, tone, and the entire app's visual colors** dynamically based on the emotion of the conversation.

<p align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWJ5ZDRibmxpZzhkZWVxNDE0M2E4MzMwNXE2b2ZrcGoxdWg5YWtyOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l3vR1v87J9R19sLgk/giphy.gif" alt="AI Ambient Light" width="100%" style="border-radius:24px; box-shadow: 0 0 30px rgba(0,0,0,0.5);" />
</p>

## 🚀 Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🧠 Emotional Intelligence</h3>
      <p>Aria implements internal arc tracking. If you are sad, the UI turns a soft indigo, and her voice lowers in pitch and pace. If you are excited, the UI glows pink and she speaks enthusiastically.</p>
    </td>
    <td width="50%">
      <h3>⚡ Blazing Fast Architecture</h3>
      <p>Achieves sub-second vocal response latency using Groq's LPU layer and direct WebSocket <code>ArrayBuffer</code> streaming straight to the browser's audio context.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🌍 Seamless Bilingualism</h3>
      <p>Switch between a native 🇺🇸 <b>English</b> persona and a natural 🇸🇦 <b>Arabic (Saudi)</b> persona instantly. The UI mirrors this by automatically transitioning from LTR to RTL formatting.</p>
    </td>
    <td width="50%">
      <h3>📱 Native Telegram App</h3>
      <p>Aria lives outside the browser too! A fully decoupled async Telegram Bot (<code>telegram_bot.py</code>) allows you to send her text or voice messages on the go and receive high-fidelity voice notes back.</p>
    </td>
  </tr>
</table>

## 🎨 The "Old Money" Aesthetic 

Aria features a **neo-brutalist / old-money** UI inspired by luxury fluid design:
- 🚫 **Zero Bloat:** No React, No Tailwind. Pure Vanilla CSS3 mastery.
- 🌌 **Ambient Lighting:** Generative, shape-shifting localized glows mimic Aria's core engine breathing in the background.
- ✨ **Haptic Visuals:** Elements pop and fade seamlessly with 60fps glass-morphism animations.

---

## ⚙️ Getting Started

### 1️⃣ Clone & Install
```bash
git clone https://github.com/mohasbks/Aria-Voice-Companion.git
cd Aria-Voice-Companion/backend
pip install -r requirements.txt
```

### 2️⃣ Environment Variables (`.env`)
You retain total control over your API keys (Keys can also be injected securely via the Frontend Interface):
```env
GROQ_API_KEY=gsk_your_english_key_here
GROQ_API_KEY_ARABIC=gsk_your_arabic_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### 3️⃣ Ignite the Engines 🚀
Fire up the **FastAPI Web Server**:
```bash
cd backend
python main.py
```
Fire up the **Telegram Bot** (optional, in a new terminal):
```bash
python telegram_bot.py
```
> **Access the App:** Open `http://localhost:8000` in your web browser.

---

## 🔮 Roadmap / Next Steps
- [ ] Implement VAD (Voice Activity Detection) for interruption handling.
- [ ] Support LLM Vision (Image sharing on Telegram).
- [ ] Expanded Memory via Vector DB for long-term recall.

<br>
<h3 align="center">Made by a developer, for developers. Enjoy solving the silence. 🌙</h3>
