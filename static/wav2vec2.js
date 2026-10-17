/* 语音情感识别 · wav2vec2 前端交互（录音/上传/波形播放/概率） */
(function () {
  'use strict';

  const labels = window.SER_LABELS;
  const labelsZh = window.SER_LABELS_ZH;
  /* ★ 改动：读取情感 → emoji 映射，用于标签和主结果展示 */
  const labelsEmoji = window.SER_LABELS_EMOJI || {};
  const emotionColors = ['--emotion-0', '--emotion-1', '--emotion-2', '--emotion-3'];
  const MAX_RECORD_SEC = 8;
  const COLOR_MAIN = '#0f766e';   // 已播放部分
  const COLOR_REST = 'rgba(23,25,28,0.28)'; // 未播放部分

  let mediaRecorder = null;
  let audioChunks = [];
  let recordTimer = null;
  let currentAudioBuffer = null;
  let currentAudioBlob = null;
  let audioPlayer = null;
  let peaks = [];            // 预计算的波形峰值（按列）
  let rafId = null;

  const recordBtn = document.getElementById('record-btn');
  const recordCard = document.getElementById('record-card');
  const recordDesc = document.getElementById('record-desc');
  const uploadBtn = document.getElementById('upload-btn');
  const uploadCard = document.getElementById('upload-card');
  const uploadDesc = document.getElementById('upload-desc');
  const fileInput = document.getElementById('file-input');
  const wavePanel = document.getElementById('wave-panel');
  const emptyPanel = document.getElementById('empty-panel');
  const waveform = document.getElementById('waveform');
  const playBtn = document.getElementById('play-btn');
  const playIco = document.getElementById('play-ico');
  const waveMeta = document.getElementById('wave-meta');
  const resultPanel = document.getElementById('result-panel');
  const resultEmotion = document.getElementById('result-emotion');
  const resultSub = document.getElementById('result-sub');
  const eq = document.getElementById('eq');
  const status = document.getElementById('status');

  /* ---- 入场动画 ---- */
  document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in'));

  /* ---- 构建声柱 ---- */
  function buildEq() {
    eq.innerHTML = '';
    labels.forEach((label, i) => {
      const bar = document.createElement('div');
      bar.className = 'eq-bar';
      bar.innerHTML =
        '<span class="eq-value" data-idx="' + i + '">0%</span>' +
        '<div class="eq-track"><div class="eq-fill" data-idx="' + i +
        '" style="height:0%;background:var(' + emotionColors[i] + ')"></div></div>' +
        /* ★ 改动：在情感标签前加 emoji 表情 */
        '<span class="eq-label">' + (labelsEmoji[label] || '') + ' ' + label + '<br>' + labelsZh[label] + '</span>';
      eq.appendChild(bar);
    });
  }

  function updateEq(probabilities) {
    labels.forEach((_, i) => {
      const pct = Math.round(probabilities[i] * 100);
      const fill = eq.querySelector('.eq-fill[data-idx="' + i + '"]');
      const val = eq.querySelector('.eq-value[data-idx="' + i + '"]');
      if (fill) fill.style.height = pct + '%';
      if (val) val.textContent = pct + '%';
    });
  }

  function setStatus(text, busy) {
    status.innerHTML = '<span class="status-dot"' +
      (busy ? ' style="background:var(--emotion-1)"' : '') + '></span>' + text;
  }

  /* ---- 波形：预计算峰值 ---- */
  function computePeaks(audioBuffer, columns) {
    const data = audioBuffer.getChannelData(0);
    const step = Math.max(1, Math.floor(data.length / columns));
    const arr = [];
    for (let i = 0; i < columns; i++) {
      const start = i * step;
      let peak = 0;
      for (let j = 0; j < step && start + j < data.length; j++) {
        const v = Math.abs(data[start + j]);
        if (v > peak) peak = v;
      }
      arr.push(peak);
    }
    const max = Math.max.apply(null, arr) || 1;
    return arr.map((p) => p / max); // 归一化，避免小音量波形看不出来
  }

  /* ---- 波形绘制（progress: 0~1，已播放部分高亮）---- */
  function drawWave(progress) {
    const dpr = window.devicePixelRatio || 1;
    const w = waveform.clientWidth || waveform.offsetWidth || 600;
    const h = 150;
    waveform.width = w * dpr;
    waveform.height = h * dpr;
    waveform.style.height = h + 'px';
    const ctx = waveform.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const n = peaks.length;
    const mid = h / 2;
    for (let i = 0; i < w; i++) {
      const idx = Math.floor(i / w * n) || 0;
      const amp = peaks[idx] || 0;
      const barH = Math.max(1, amp * (mid - 6));
      ctx.fillStyle = (i / w) <= progress ? COLOR_MAIN : COLOR_REST;
      ctx.fillRect(i, mid - barH, 1, barH * 2);
    }
    // 进度线
    if (progress > 0 && progress < 1) {
      ctx.fillStyle = 'rgba(23,25,28,0.5)';
      ctx.fillRect(Math.floor(w * progress), mid - (mid - 6), 1.5, (mid - 6) * 2);
    }
  }

  function showWave(audioBuffer) {
    wavePanel.hidden = false;               // 先显示面板，保证能取到宽度
    if (emptyPanel) emptyPanel.hidden = true;
    requestAnimationFrame(() => {
      const w = waveform.clientWidth || 600;
      peaks = computePeaks(audioBuffer, w);
      drawWave(0);
      waveMeta.innerHTML = '<span>时长 ' + audioBuffer.duration.toFixed(2) + ' 秒</span>' +
        '<span>采样率 ' + audioBuffer.sampleRate + ' Hz → 16 kHz</span>';
    });
  }

  /* ---- 播放（带进度染色）---- */
  function setPlayIcon(playing) {
    playIco.textContent = playing ? '❚❚' : '▶';
    playBtn.classList.toggle('playing', playing);
    playBtn.setAttribute('aria-label', playing ? '停止播放' : '播放音频');
  }

  function stopProgressLoop() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  }

  function startProgressLoop() {
    stopProgressLoop();
    const tick = () => {
      if (!audioPlayer) return;
      const p = audioPlayer.duration
        ? Math.min(1, audioPlayer.currentTime / audioPlayer.duration) : 0;
      drawWave(p);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  function pausePlayback() {
    if (audioPlayer) audioPlayer.pause();
    setPlayIcon(false);
    stopProgressLoop();
  }

  function togglePlay() {
    if (!currentAudioBlob) return;
    if (audioPlayer && !audioPlayer.paused) {   // 正在播 → 停止
      pausePlayback();
      drawWave(0);
      return;
    }
    if (audioPlayer) { audioPlayer.pause(); }
    audioPlayer = new Audio(URL.createObjectURL(currentAudioBlob));
    audioPlayer.onended = () => {
      setPlayIcon(false);
      stopProgressLoop();
      drawWave(0);
    };
    audioPlayer.play();
    setPlayIcon(true);
    startProgressLoop();
  }

  /* ---- 重采样到 16kHz 并转 WAV ---- */
  async function audioBufferToWav16k(audioBuffer) {
    const targetRate = 16000;
    const offline = new OfflineAudioContext(
      1, Math.ceil(audioBuffer.duration * targetRate), targetRate);
    const src = offline.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(offline.destination);
    src.start();
    const rendered = await offline.startRendering();
    const pcm = rendered.getChannelData(0);
    return encodeWav(pcm, targetRate);
  }

  function encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  /* ---- 预测 ---- */
  async function predict(wavBlob) {
    setStatus('识别中 PROCESSING', true);
    const formData = new FormData();
    formData.append('file', wavBlob, 'audio.wav');
    try {
      const resp = await fetch('/predict', { method: 'POST', body: formData });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || '预测失败');
      /* ★ 改动：主结果前加对应情感 emoji（如 😊 happy） */
      resultEmotion.textContent = (labelsEmoji[data.emotion] || '') + ' ' + data.emotion;
      resultEmotion.style.color = 'var(' + emotionColors[data.emotion_idx] + ')';
      resultSub.textContent = data.emotion_zh + ' · 置信度 ' +
        Math.round(data.probabilities[data.emotion_idx] * 100) + '% · 时长 ' + data.duration + ' 秒';
      updateEq(data.probabilities);
      resultPanel.hidden = false;
      setStatus('完成 DONE', false);
    } catch (err) {
      alert('识别失败：' + err.message);
      setStatus('出错 ERROR', false);
    }
  }

  /* ---- 处理音频（解码 → 画波形 → 预测）---- */
  async function processAudioBlob(blob) {
    try {
      pausePlayback();
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      currentAudioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      currentAudioBlob = blob;
      if (playBtn) playBtn.disabled = false;
      showWave(currentAudioBuffer);
      const wavBlob = await audioBufferToWav16k(currentAudioBuffer);
      await predict(wavBlob);
    } catch (err) {
      alert('音频处理失败：' + err.message);
      setStatus('出错 ERROR', false);
    }
  }

  /* ---- 录音 ---- */
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        processAudioBlob(blob);
      };
      mediaRecorder.start();
      recordBtn.classList.add('recording');
      recordBtn.innerHTML = '<span class="dot"></span>停止录音';
      recordCard.classList.add('active');
      recordDesc.textContent = '正在录音…';
      setStatus('录音中 RECORDING', true);
      recordTimer = setTimeout(() => {
        if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
      }, MAX_RECORD_SEC * 1000);
    } catch (err) {
      alert('无法访问麦克风：' + err.message);
    }
  }

  function stopRecording() {
    if (recordTimer) { clearTimeout(recordTimer); recordTimer = null; }
    if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
    recordBtn.classList.remove('recording');
    recordBtn.innerHTML = '<span class="dot"></span>开始录音';
    recordCard.classList.remove('active');
    recordDesc.textContent = '点击开始录制，最长 8 秒';
  }

  /* ---- 事件绑定 ---- */
  recordBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
    else startRecording();
  });

  uploadBtn.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadDesc.textContent = file.name;
    uploadCard.classList.add('active');
    processAudioBlob(file);
  });

  if (playBtn) playBtn.addEventListener('click', togglePlay);

  buildEq();
})();
