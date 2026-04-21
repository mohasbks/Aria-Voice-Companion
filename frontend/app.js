/**
 * app.js – Aria Voice Chat Frontend
 *
 * Handles:
 *  - WebSocket lifecycle (connect / reconnect)
 *  - Dual input modes:
 *      1. Browser mode  → Web Speech API transcribes locally, sends text over WS
 *      2. Groq STT mode → MediaRecorder captures audio, sends binary blob over WS
 *  - Incoming WS message routing (transcript / response / audio / error)
 *  - Streaming MP3 playback via MSE (MediaSource) or fallback Blob URL
 *  - Chat UI rendering with emotion badges
 *  - Toast notification system
 *  - Conversation memory (stored in sessionStorage for display; real memory in SQLite)
 */

"use strict";

/* ══════════════════════════════════════════════════════════════
   CONFIG
   ══════════════════════════════════════════════════════════════ */
const WS_URL        = `ws://${location.host}/ws`;
const RECONNECT_MS  = 3000;   // Reconnect delay
const MAX_RECONNECTS = 10;

/* ══════════════════════════════════════════════════════════════
   STATE
   ══════════════════════════════════════════════════════════════ */
let ws            = null;
let reconnectCount = 0;
let isConnected   = false;

// Session
const sessionId   = `session_${Date.now()}_${Math.random().toString(36).slice(2,7)}`;

// Input mode: 'webspeech' | 'groq'
let inputMode     = 'webspeech';

// Recording state
let mediaRecorder = null;
let audioChunks   = [];
let isRecording   = false;

// Web Speech API
let recognition   = null;
let finalTranscript = '';
let interimTranscript = '';

// Audio playback via MSE stream
let mediaSource   = null;
let sourceBuffer  = null;
let audioQueue    = [];        // queued binary chunks
let isPlaying     = false;
let currentAudioBlob = null;   // for fallback
let audioChunksForBlob = [];

/* ══════════════════════════════════════════════════════════════
   DOM REFERENCES
   ══════════════════════════════════════════════════════════════ */
const chatWindow      = document.getElementById('chat-window');
const welcomeCard     = document.getElementById('welcome-card');
const micBtn          = document.getElementById('mic-btn');
const iconMic         = document.getElementById('icon-mic');
const iconStop        = document.getElementById('icon-stop');
const statusDot       = document.getElementById('status-dot');
const statusLabel     = document.getElementById('status-label');
const sessionDisplay  = document.getElementById('session-display');
const interimText     = document.getElementById('interim-text');
const transcriptPill  = document.getElementById('transcript-pill');
const activityInner   = document.getElementById('activity-inner');
const activityText    = document.getElementById('activity-text');
const textInput       = document.getElementById('text-input');
const sendBtn         = document.getElementById('send-btn');
const clearBtn        = document.getElementById('clear-btn');
const settingsBtn     = document.getElementById('settings-btn');
const aiAudio         = document.getElementById('ai-audio');
const toastContainer  = document.getElementById('toast-container');
const btnWebSpeech    = document.getElementById('btn-webspeech');
const btnGroq         = document.getElementById('btn-groq');
const waveBars        = document.getElementById('wave-bars');

/* User Settings */
let userLang = localStorage.getItem('aria_lang') || 'en';
let userGroqKey = localStorage.getItem('aria_groq_key') || '';
let userGroqKeyAr = localStorage.getItem('aria_groq_key_ar') || '';

/* ══════════════════════════════════════════════════════════════
   WEBSOCKET
   ══════════════════════════════════════════════════════════════ */

function connectWS() {
  setStatus('connecting', 'Connecting…');
  ws = new WebSocket(WS_URL);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    isConnected = true;
    reconnectCount = 0;
    setStatus('connected', 'Connected');
    // Send init frame so the server knows our session
    wsSend({ 
      type: 'init', 
      session_id: sessionId, 
      mime_type: getMimeType(),
      lang: userLang,
      groq_key: userGroqKey,
      groq_key_ar: userGroqKeyAr
    });
    sessionDisplay.textContent = sessionId.slice(-8);
  };

  ws.onclose = () => {
    isConnected = false;
    setStatus('disconnected', 'Disconnected');
    scheduleReconnect();
  };

  ws.onerror = () => {
    isConnected = false;
    setStatus('disconnected', 'Connection error');
  };

  ws.onmessage = handleWSMessage;
}

function scheduleReconnect() {
  if (reconnectCount >= MAX_RECONNECTS) {
    showToast('Cannot reconnect to server. Please refresh the page.', 'error', 8000);
    return;
  }
  reconnectCount++;
  setTimeout(connectWS, RECONNECT_MS);
}

function wsSend(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(typeof data === 'string' ? data : JSON.stringify(data));
  }
}

/* ══════════════════════════════════════════════════════════════
   INCOMING MESSAGE ROUTING
   ══════════════════════════════════════════════════════════════ */

// Buffers binary audio frames while we assemble the full response
let binaryChunks  = [];
let currentEmotion = 'calm';

function handleWSMessage(event) {
  // Binary frame → audio data
  if (event.data instanceof ArrayBuffer) {
    binaryChunks.push(new Uint8Array(event.data));
    return;
  }

  let msg;
  try { msg = JSON.parse(event.data); }
  catch { return; }

  switch (msg.type) {
    case 'ready':
      break;

    case 'processing':
      showActivity(true, stageLabel(msg.stage));
      break;

    case 'transcript':
      showActivity(false);
      interimText.textContent = '';
      transcriptPill.classList.remove('active');
      appendMessage('user', msg.text);
      break;

    case 'response':
      appendMessage('assistant', msg.text, msg.emotion);
      if (msg.arc && msg.arc.length) updateEmotionArc(msg.arc);
      else if (msg.emotion) applyEmotionTheme(msg.emotion); // fallback if no arc yet
      showActivity(true, '🔊 Generating voice…');
      currentEmotion = msg.emotion || 'calm';
      binaryChunks = []; // reset audio buffer
      break;

    case 'audio_start':
      binaryChunks = [];
      break;

    case 'audio_end':
      showActivity(false);
      playCollectedAudio(binaryChunks, currentEmotion);
      binaryChunks = [];
      break;

    case 'error':
      showActivity(false);
      showToast(`Error: ${msg.message}`, 'error');
      break;

    case 'pong':
      break;

    default:
      break;
  }
}

function stageLabel(stage) {
  const map = {
    stt: '🎙 Transcribing speech…',
    llm: '🧠 Thinking contextually…',
    tts: '🔊 Synthesizing voice…'
  };
  return map[stage] || 'Processing…';
}

/* Update the sidebar emotion arc dots */
function updateEmotionArc(arc) {
  const arcEl = document.getElementById('emotion-arc-dots');
  if (!arcEl) return;
  const icons = {
    happy:'😊', excited:'🎉', calm:'😌', sad:'😢', serious:'🎯',
    angry:'😤', playful:'😄', curious:'🤔', surprised:'😲'
  };
  arcEl.innerHTML = arc.map(e =>
    `<span class="arc-dot emo-dot emo-${e}" title="${e}">${icons[e]||'💬'}</span>`
  ).join('');

  // Apply theme based on latest emotion
  if (arc.length > 0) applyEmotionTheme(arc[arc.length - 1]);
}

/* ── Emotion Color Engine ─────────────────────────────────────
   Maps each emotion to a CSS HSL hue, which drives --accent
   throughout the entire UI. Pure CSS variable swap = smooth
   1.5s transition across ALL elements simultaneously.
   ──────────────────────────────────────────────────────────── */
const EMOTION_HUE = {
  happy:     48,   // warm gold
  excited:   320,  // hot pink
  calm:      200,  // sky blue
  sad:       240,  // indigo
  serious:   210,  // slate blue
  angry:     0,    // red
  playful:   150,  // emerald green
  curious:   30,   // orange
  surprised: 275,  // violet
};
const EMOTION_ICONS = {
  happy:'😊', excited:'🎉', calm:'😌', sad:'😢', serious:'🎯',
  angry:'😤', playful:'😄', curious:'🤔', surprised:'😲'
};

function applyEmotionTheme(emotion) {
  const hue = EMOTION_HUE[emotion] ?? 260;
  const root = document.documentElement;
  root.style.setProperty('--emotion-h', hue);

  // Update live emotion panel
  const icon = document.getElementById('live-emotion-icon');
  const name = document.getElementById('live-emotion-name');
  if (icon) icon.textContent = EMOTION_ICONS[emotion] || '💬';
  if (name) name.textContent = emotion;
}

/* ══════════════════════════════════════════════════════════════
   AUDIO PLAYBACK
   ══════════════════════════════════════════════════════════════ */

function playCollectedAudio(chunks, emotion) {
  if (!chunks || chunks.length === 0) return;

  // Merge all chunks into a single Uint8Array
  const totalLen = chunks.reduce((s, c) => s + c.length, 0);
  const merged = new Uint8Array(totalLen);
  let offset = 0;
  for (const c of chunks) { merged.set(c, offset); offset += c.length; }

  // Auto-detect format: WAV starts with "RIFF"; otherwise assume MP3
  let mimeType = 'audio/mpeg';
  if (merged[0] === 0x52 && merged[1] === 0x49 && merged[2] === 0x46 && merged[3] === 0x46) {
    mimeType = 'audio/wav';  // Orpheus / Kokoro output
  }

  const blob = new Blob([merged], { type: mimeType });
  const url  = URL.createObjectURL(blob);

  aiAudio.pause();
  aiAudio.src = url;

  // Show speaking waveform on last assistant message
  const lastMsg = document.querySelector('.message.assistant:last-child');
  if (lastMsg) {
    lastMsg.classList.add('speaking');
    let wave = lastMsg.querySelector('.aria-speaking-wave');
    if (!wave) {
      wave = document.createElement('div');
      wave.className = `aria-speaking-wave emotion-${emotion || 'calm'}`;
      wave.innerHTML = '<span></span>'.repeat(7);
      const avatar = lastMsg.querySelector('.msg-avatar');
      if (avatar) avatar.after(wave);
    }
    wave.classList.add('active');

    aiAudio.onended = () => {
      URL.revokeObjectURL(url);
      lastMsg.classList.remove('speaking');
      wave.classList.remove('active');
    };
  } else {
    aiAudio.onended = () => URL.revokeObjectURL(url);
  }

  aiAudio.play().catch(e => console.warn('Audio play blocked:', e));
}

/* ══════════════════════════════════════════════════════════════
   MIC BUTTON – MODE DISPATCHER
   ══════════════════════════════════════════════════════════════ */

micBtn.addEventListener('click', () => {
  if (inputMode === 'webspeech') {
    toggleWebSpeech();
  } else {
    toggleMediaRecorder();
  }
});

/* ══════════════════════════════════════════════════════════════
   MODE 1: Web Speech API (browser STT)
   ══════════════════════════════════════════════════════════════ */

function initWebSpeech() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showToast('Web Speech API not supported in this browser. Switch to Groq STT mode.', 'error', 6000);
    return null;
  }

  const rec = new SpeechRecognition();
  rec.continuous      = false;
  rec.interimResults  = true;
  rec.lang            = 'en-US';
  rec.maxAlternatives = 1;

  rec.onstart = () => {
    isRecording = true;
    setMicState(true);
    showActivity(true, '🎙 Listening…');
    transcriptPill.classList.add('active');
    finalTranscript   = '';
    interimTranscript = '';
  };

  rec.onresult = (e) => {
    let interim = '';
    let final   = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) final += t;
      else interim += t;
    }
    finalTranscript   += final;
    interimTranscript  = interim;
    interimText.textContent = (finalTranscript + ' ' + interimTranscript).trim();
  };

  rec.onend = () => {
    isRecording = false;
    setMicState(false);
    transcriptPill.classList.remove('active');
    interimText.textContent = '';

    const text = finalTranscript.trim();
    if (text) {
      wsSend({ type: 'text', text, session_id: sessionId });
      showActivity(true, '🤔 Thinking…');
    } else {
      showActivity(false);
    }
  };

  rec.onerror = (e) => {
    console.warn('Speech recognition error:', e.error);
    isRecording = false;
    setMicState(false);
    showActivity(false);
    if (e.error !== 'no-speech') {
      showToast(`Speech error: ${e.error}`, 'error');
    }
  };

  return rec;
}

function toggleWebSpeech() {
  if (!isConnected) { showToast('Not connected to server.', 'error'); return; }

  if (isRecording) {
    recognition && recognition.stop();
  } else {
    recognition = initWebSpeech();
    if (recognition) recognition.start();
  }
}

/* ══════════════════════════════════════════════════════════════
   MODE 2: MediaRecorder → Groq Whisper
   ══════════════════════════════════════════════════════════════ */

async function toggleMediaRecorder() {
  if (!isConnected) { showToast('Not connected to server.', 'error'); return; }

  if (isRecording) {
    // Stop recording – ondataavailable fires, then onstop sends the blob
    mediaRecorder && mediaRecorder.stop();
    isRecording = false;
    setMicState(false);
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = getMimeType();

    mediaRecorder = new MediaRecorder(stream, { mimeType });
    audioChunks   = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(audioChunks, { type: mimeType });
      if (blob.size < 500) {
        showToast('Audio too short. Please try again.', 'info');
        return;
      }
      const buffer = await blob.arrayBuffer();
      if (ws && ws.readyState === WebSocket.OPEN) {
        showActivity(true, '🎙 Transcribing…');
        ws.send(buffer);
      }
    };

    mediaRecorder.start(250); // collect in 250ms chunks
    isRecording = true;
    setMicState(true);
    showActivity(true, '🎙 Recording…');
    transcriptPill.classList.add('active');
    interimText.textContent = '● Recording…';
  } catch (err) {
    showToast('Microphone access denied: ' + err.message, 'error');
  }
}

function getMimeType() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) return t;
  }
  return 'audio/webm';
}

/* ══════════════════════════════════════════════════════════════
   TEXT INPUT
   ══════════════════════════════════════════════════════════════ */

function sendText() {
  const text = textInput.value.trim();
  if (!text) return;
  if (!isConnected) { showToast('Not connected to server.', 'error'); return; }

  textInput.value = '';
  wsSend({ type: 'text', text, session_id: sessionId });
  showActivity(true, '🤔 Thinking…');
}

sendBtn.addEventListener('click', sendText);
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText(); }
});

/* ══════════════════════════════════════════════════════════════
   MODE TOGGLE
   ══════════════════════════════════════════════════════════════ */

document.getElementById('mode-group').addEventListener('click', (e) => {
  const btn = e.target.closest('.toggle-btn');
  if (!btn) return;
  const mode = btn.dataset.mode;
  if (mode === inputMode) return;

  inputMode = mode;
  btnWebSpeech.classList.toggle('active', mode === 'webspeech');
  btnGroq.classList.toggle('active', mode === 'groq');

  // Stop any active recording when switching
  if (isRecording) {
    if (recognition) recognition.stop();
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    isRecording = false;
    setMicState(false);
    showActivity(false);
  }

  showToast(mode === 'webspeech'
    ? 'Browser STT mode: transcript happens in your browser'
    : 'Groq STT mode: audio sent to Groq Whisper', 'info', 3000);
});

/* ══════════════════════════════════════════════════════════════
   CLEAR CHAT
   ══════════════════════════════════════════════════════════════ */

clearBtn.addEventListener('click', () => {
  // Remove all messages except the welcome card
  const messages = chatWindow.querySelectorAll('.message');
  messages.forEach(m => m.remove());
  welcomeCard.style.display = '';
  showToast('Chat cleared', 'success', 2000);
});

/* ══════════════════════════════════════════════════════════════
   CHAT UI HELPERS
   ══════════════════════════════════════════════════════════════ */

function appendMessage(role, text, emotion = null) {
  // Hide welcome card on first message
  if (welcomeCard) welcomeCard.style.display = 'none';

  const now  = new Date();
  const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const div = document.createElement('div');
  const dirClass = userLang === 'ar' ? 'rtl' : 'ltr';
  div.className = `message ${role} ${dirClass}`;

  const avatar = role === 'user' ? '🧑' : '🤖';
  const name   = role === 'user' ? 'You' : 'Aria';

  let emotionBadge = '';
  if (emotion) {
    emotionBadge = `<span class="msg-emotion-badge emotion-${emotion}">${emotionIcon(emotion)} ${emotion}</span>`;
  }

  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-meta">
        <span class="msg-name">${name}</span>
        <span class="msg-time">${time}</span>
        ${emotionBadge}
      </div>
      <div class="msg-bubble">${escapeHtml(text)}</div>
    </div>
  `;

  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function emotionIcon(emotion) {
  const map = {
    happy:     '😊',
    excited:   '🎉',
    calm:      '😌',
    sad:       '😢',
    serious:   '🎯',
    angry:     '😤',
    playful:   '😄',
    curious:   '🤔',
    surprised: '😲',
  };
  return map[emotion] || '💬';
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/* ══════════════════════════════════════════════════════════════
   UI STATE HELPERS
   ══════════════════════════════════════════════════════════════ */

function setMicState(recording) {
  micBtn.classList.toggle('recording', recording);
  micBtn.setAttribute('aria-pressed', recording.toString());
  iconMic.style.display  = recording ? 'none'  : '';
  iconStop.style.display = recording ? ''       : 'none';
  micBtn.title = recording ? 'Stop recording' : 'Start recording';
}

function setStatus(state, label) {
  statusDot.className = `status-dot ${state}`;
  statusLabel.textContent = label;
}

function showActivity(visible, text = '') {
  activityText.textContent = text;
  activityInner.classList.toggle('visible', visible);
  waveBars.style.display = visible ? '' : 'none';
}

/* ══════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
   ══════════════════════════════════════════════════════════════ */

function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.4s ease';
    setTimeout(() => toast.remove(), 400);
  }, duration);
}

/* ══════════════════════════════════════════════════════════════
   WELCOME CARD TIPS – CLICK TO SEND
   ══════════════════════════════════════════════════════════════ */

document.querySelectorAll('.tip').forEach(tip => {
  tip.addEventListener('click', () => {
    const text = tip.textContent.replace(/^Try:\s*/i, '').replace(/"/g, '').trim();
    textInput.value = text;
    sendText();
  });
});

/* ══════════════════════════════════════════════════════════════
   SETTINGS MODAL LOGIC
   ══════════════════════════════════════════════════════════════ */

const settingsModal = document.getElementById('settings-modal');
const closeSettingsBtn = document.getElementById('close-settings-btn');
const saveSettingsBtn = document.getElementById('save-settings-btn');
const groqKeyInput = document.getElementById('groq-key-input');
const groqKeyArInput = document.getElementById('groq-key-ar-input');
const langBtns = document.querySelectorAll('.lang-btn');

function applyLangSelection(lang) {
  langBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });
}

function updatePlaceholderByLang() {
  if (userLang === "ar") {
    textInput.placeholder = "تحدث مع آريا...";
    textInput.dir = "rtl";
  } else {
    textInput.placeholder = "Message Aria...";
    textInput.dir = "ltr";
  }
}

settingsBtn.addEventListener('click', () => {
  /* Load current values */
  groqKeyInput.value = userGroqKey;
  groqKeyArInput.value = userGroqKeyAr;
  applyLangSelection(userLang);
  settingsModal.classList.add('active');
});

closeSettingsBtn.addEventListener('click', () => {
  settingsModal.classList.remove('active');
});

langBtns.forEach(btn => {
  btn.addEventListener('click', (e) => {
    applyLangSelection(btn.dataset.lang);
  });
});

saveSettingsBtn.addEventListener('click', () => {
  const selectedLang = Array.from(langBtns).find(b => b.classList.contains('active')).dataset.lang;
  const key1 = groqKeyInput.value.trim();
  const key2 = groqKeyArInput.value.trim();

  // Save to local state
  userLang = selectedLang;
  userGroqKey = key1;
  userGroqKeyAr = key2;

  // Persist
  localStorage.setItem('aria_lang', userLang);
  localStorage.setItem('aria_groq_key', userGroqKey);
  localStorage.setItem('aria_groq_key_ar', userGroqKeyAr);

  // Tell backend live
  wsSend({
    type: 'settings',
    lang: userLang,
    groq_key: userGroqKey,
    groq_key_ar: userGroqKeyAr
  });

  updatePlaceholderByLang();
  showToast('Settings saved successfully.', 'success');
  settingsModal.classList.remove('active');
});

/* ══════════════════════════════════════════════════════════════
   KEEPALIVE PING
   ══════════════════════════════════════════════════════════════ */

setInterval(() => {
  if (isConnected) wsSend({ type: 'ping' });
}, 25000);

/* ══════════════════════════════════════════════════════════════
   BOOT
   ══════════════════════════════════════════════════════════════ */

updatePlaceholderByLang();
connectWS();
