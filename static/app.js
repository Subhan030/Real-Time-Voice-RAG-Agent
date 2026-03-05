// ============================================================
// Voice RAG Agent — Client-Side Logic
// WebSocket + Audio Recording + Barge-In + Continuous Mode
// ============================================================

const SILENCE_THRESHOLD = 0.015;
const SILENCE_DURATION_MS = 1500;
const BARGE_IN_THRESHOLD = 0.04;
const BARGE_IN_SUSTAINED_MS = 300;

const micBtn = document.getElementById('mic-btn');
const micIcon = document.querySelector('.mic-icon');
const stopIcon = document.querySelector('.stop-icon');
const micHint = document.getElementById('mic-hint');
const chatMessages = document.getElementById('chat-messages');
const chatContainer = document.getElementById('chat-container');
const statusText = document.getElementById('status-text');
const statusBar = document.getElementById('status-bar');
const connDot = document.getElementById('connection-dot');
const fileUpload = document.getElementById('file-upload');
const uploadToast = document.getElementById('upload-toast');
const toastMessage = document.getElementById('toast-message');
const welcomeMessage = document.getElementById('welcome-message');

let ws = null;
let mediaRecorder = null;
let audioContext = null;
let analyser = null;
let micStream = null;
let isRecording = false;
let audioChunks = [];
let silenceTimer = null;
let currentAgentMessage = null;
let audioQueue = [];
let isPlayingAudio = false;
let bargeInMonitorId = null;
let isContinuous = false;

// ============================================================
// WebSocket
// ============================================================

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        connDot.classList.add('connected');
        connDot.title = 'Connected';
        setStatus('Ready — click the mic to speak');
    };

    ws.onclose = () => {
        connDot.classList.remove('connected');
        connDot.title = 'Disconnected';
        setStatus('Disconnected — reconnecting...');
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => console.error('[WS] Error');

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        handleServerMessage(msg);
    };
}

function handleServerMessage(msg) {
    switch (msg.type) {
        case 'status': handleStatusUpdate(msg.state); break;
        case 'transcript': addUserMessage(msg.text); break;
        case 'answer_text': handleAnswerText(msg.text, msg.done); break;
        case 'answer_audio': handleAnswerAudio(msg.audio, msg.format); break;
        case 'error': setStatus(`Error: ${msg.message}`); break;
    }
}

function handleStatusUpdate(state) {
    switch (state) {
        case 'listening':
            setStatus('Ready — click the mic to speak');
            micBtn.disabled = false;
            checkContinuousResume();
            break;
        case 'processing':
            setStatus('Processing your question...');
            statusBar.classList.add('active');
            showTypingIndicator();
            break;
        case 'speaking':
            setStatus('Speaking...');
            break;
    }
}

// ============================================================
// Chat Messages
// ============================================================

function hideWelcome() {
    if (welcomeMessage) welcomeMessage.style.display = 'none';
}

function addUserMessage(text) {
    hideWelcome();
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `<div class="message-avatar">🧑</div><div class="message-content">${escapeHtml(text)}</div>`;
    removeTypingIndicator();
    chatMessages.appendChild(div);
    scrollToBottom();
}

function showTypingIndicator() {
    removeTypingIndicator();
    hideWelcome();
    const div = document.createElement('div');
    div.className = 'message agent';
    div.id = 'typing-indicator';
    div.innerHTML = `<div class="message-avatar">🤖</div><div class="message-content"><div class="typing-dots"><span></span><span></span><span></span></div></div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function handleAnswerText(text, done) {
    removeTypingIndicator();

    if (!currentAgentMessage && text) {
        const div = document.createElement('div');
        div.className = 'message agent';
        div.innerHTML = `<div class="message-avatar">🤖</div><div class="message-content"><div class="agent-text"></div></div>`;
        chatMessages.appendChild(div);
        currentAgentMessage = div;
    }

    if (text && currentAgentMessage) {
        const textEl = currentAgentMessage.querySelector('.agent-text');
        textEl.textContent = textEl.textContent ? textEl.textContent + ' ' + text : text;
        scrollToBottom();
    }

    if (done) {
        currentAgentMessage = null;
        setStatus('Ready — click the mic to speak');
        statusBar.classList.remove('active');
    }
}

function handleAnswerAudio(audioB64, format) {
    const bytes = base64ToArrayBuffer(audioB64);
    const mimeType = format === 'mp3' ? 'audio/mpeg' : 'audio/wav';
    const url = URL.createObjectURL(new Blob([bytes], { type: mimeType }));
    audioQueue.push(url);
    playNextInQueue();
}

// ============================================================
// Audio Playback Queue + Barge-In
// ============================================================

function playNextInQueue() {
    if (isPlayingAudio || audioQueue.length === 0) return;

    isPlayingAudio = true;
    const url = audioQueue.shift();
    const audio = new Audio(url);
    window._currentAudio = audio;

    startBargeInMonitor();

    const onDone = () => {
        URL.revokeObjectURL(url);
        isPlayingAudio = false;
        if (audioQueue.length === 0) {
            stopBargeInMonitor();
            setStatus('Ready — click the mic to speak');
            statusBar.classList.remove('active');
            checkContinuousResume();
        } else {
            playNextInQueue();
        }
    };

    audio.onended = onDone;
    audio.onerror = onDone;
    audio.play().catch(onDone);
}

function stopAllPlayback() {
    if (window._currentAudio) {
        window._currentAudio.pause();
        window._currentAudio = null;
    }
    audioQueue.forEach(url => URL.revokeObjectURL(url));
    audioQueue = [];
    isPlayingAudio = false;
    stopBargeInMonitor();
}

// ============================================================
// Continuous Mode
// ============================================================

function checkContinuousResume() {
    if (isContinuous) {
        setTimeout(() => {
            if (isContinuous && !isPlayingAudio && !isRecording && audioQueue.length === 0) {
                startRecording();
            }
        }, 500);
    }
}

// ============================================================
// Barge-In Monitor
// ============================================================

async function startBargeInMonitor() {
    if (bargeInMonitorId) return;

    try {
        if (!micStream) {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true }
            });
        }
        if (!audioContext) audioContext = new AudioContext();

        const src = audioContext.createMediaStreamSource(micStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        src.connect(analyser);

        let count = 0;
        const needed = Math.ceil(BARGE_IN_SUSTAINED_MS / 50);

        bargeInMonitorId = setInterval(() => {
            if (getVolume() > BARGE_IN_THRESHOLD) {
                if (++count >= needed) triggerBargeIn();
            } else {
                count = 0;
            }
        }, 50);
    } catch (err) {
        console.error('[Barge-in] Mic error:', err);
    }
}

function stopBargeInMonitor() {
    if (bargeInMonitorId) { clearInterval(bargeInMonitorId); bargeInMonitorId = null; }
}

function triggerBargeIn() {
    stopAllPlayback();
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'interrupt' }));
    currentAgentMessage = null;
    isContinuous = true;
    setStatus('Interrupted — listening to you...');
    startRecording();
}

// ============================================================
// Recording
// ============================================================

async function startRecording() {
    if (isRecording) return;

    try {
        if (!micStream) {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 }
            });
        }
        if (!audioContext) audioContext = new AudioContext();
        const src = audioContext.createMediaStreamSource(micStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        src.connect(analyser);

        audioChunks = [];
        mediaRecorder = new MediaRecorder(micStream, {
            mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm'
        });
        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.onstop = () => {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            sendAudioToServer(blob);
        };
        mediaRecorder.start(100);
        isRecording = true;

        micBtn.classList.add('recording');
        micIcon.style.display = 'none';
        stopIcon.style.display = 'block';
        micHint.textContent = isContinuous ? 'Listening... (auto)' : 'Listening... (click to stop)';
        setStatus('🎤 Listening...');

        startSilenceDetection();
    } catch (err) {
        console.error('[Mic] Error:', err);
        setStatus('Microphone access denied');
    }
}

function stopRecording() {
    if (!isRecording) return;
    clearTimeout(silenceTimer);
    isRecording = false;
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    micBtn.classList.remove('recording');
    micIcon.style.display = 'block';
    stopIcon.style.display = 'none';
    micHint.textContent = isContinuous ? 'Auto-listening mode ON' : 'Click to speak';
    setStatus('Processing...');
}

function startSilenceDetection() {
    let silenceStart = null;
    let speechDetected = false;
    const check = () => {
        if (!isRecording) return;
        const vol = getVolume();
        if (vol > SILENCE_THRESHOLD) { speechDetected = true; silenceStart = null; }
        else if (speechDetected) {
            if (!silenceStart) silenceStart = Date.now();
            else if (Date.now() - silenceStart > SILENCE_DURATION_MS) { stopRecording(); return; }
        }
        requestAnimationFrame(check);
    };
    requestAnimationFrame(check);
}

function getVolume() {
    if (!analyser) return 0;
    const data = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
    return Math.sqrt(sum / data.length);
}

async function sendAudioToServer(blob) {
    if (!ws || ws.readyState !== WebSocket.OPEN) { setStatus('Not connected'); return; }
    const reader = new FileReader();
    reader.onloadend = () => {
        ws.send(JSON.stringify({ type: 'audio_data', audio: reader.result.split(',')[1] }));
    };
    reader.readAsDataURL(blob);
}

// ============================================================
// File Upload
// ============================================================

fileUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    setStatus(`Uploading ${file.name}...`);
    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        showToast(`✅ ${data.message}`);
        setStatus('Ready — ask about the new document!');
    } catch (err) {
        showToast('❌ Upload failed');
    }
    fileUpload.value = '';
});

// ============================================================
// Helpers
// ============================================================

function setStatus(text) { statusText.textContent = text; }
function showToast(msg) {
    toastMessage.textContent = msg;
    uploadToast.classList.remove('hidden');
    setTimeout(() => uploadToast.classList.add('hidden'), 3000);
}
function scrollToBottom() { chatContainer.scrollTop = chatContainer.scrollHeight; }
function escapeHtml(str) { const d = document.createElement('div'); d.textContent = str; return d.innerHTML; }
function base64ToArrayBuffer(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
}

// ============================================================
// Events
// ============================================================

micBtn.addEventListener('click', () => {
    if (isRecording) {
        isContinuous = false;
        stopRecording();
    } else {
        if (isPlayingAudio) {
            stopAllPlayback();
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'interrupt' }));
            currentAgentMessage = null;
        }
        isContinuous = true;
        startRecording();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); micBtn.click(); }
});

connectWebSocket();
