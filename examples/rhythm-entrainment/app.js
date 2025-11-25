// Minimal browser demo wiring AudioWorklet (beats), Worker (Kuramoto), and UI

const state = {
  audioContext: null,
  workletNode: null,
  worker: null,
  running: false,
  bpm: 120,
  K: 1.0,
  r: 0,
  psi: 0,
};

const indicator = document.getElementById('rhythm-indicator');
const coherenceEl = document.getElementById('coherence');
const phaseEl = document.getElementById('phase');

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const inputBpm = document.getElementById('bpm');
const inputK = document.getElementById('k');
const btnHRV = document.getElementById('btn-hrv');

function applyVisualBeat(intensity) {
  indicator.style.transform = `scale(${1 + intensity * 0.3})`;
  indicator.style.opacity = String(0.5 + intensity * 0.5);
  setTimeout(() => {
    indicator.style.transform = 'scale(1)';
    indicator.style.opacity = '0.5';
  }, 100);
}

async function initAudio() {
  if (state.audioContext) return;
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: 'interactive' });
  state.audioContext = ctx;
  await ctx.audioWorklet.addModule('./timing-worklet.js');
  const node = new AudioWorkletNode(ctx, 'timing-processor', { numberOfInputs: 0, numberOfOutputs: 0 });
  node.port.onmessage = (e) => {
    if (e.data?.type === 'beat') {
      // Map coherence to intensity
      const intensity = Math.max(0.1, state.r);
      scheduleAudioBeat(e.data.time, intensity);
      queueVisualBeat(e.data.time, intensity);
    }
  };
  state.workletNode = node;
}

function scheduleAudioBeat(time, intensity) {
  const ctx = state.audioContext;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const freq = 220 + intensity * 440;
  osc.frequency.setValueAtTime(freq, time);
  gain.gain.setValueAtTime(0.25 * intensity, time);
  gain.gain.exponentialRampToValueAtTime(0.001, time + 0.08);
  osc.connect(gain).connect(ctx.destination);
  osc.start(time);
  osc.stop(time + 0.08);
}

const visualQueue = [];
function queueVisualBeat(time, intensity) { visualQueue.push({ time, intensity }); }
function processVisualQueue() {
  if (!state.audioContext) return;
  const now = state.audioContext.currentTime;
  while (visualQueue.length && visualQueue[0].time <= now) {
    const beat = visualQueue.shift();
    applyVisualBeat(beat.intensity);
  }
  requestAnimationFrame(processVisualQueue);
}

function updateMetrics() {
  coherenceEl.textContent = state.r.toFixed(2);
  phaseEl.textContent = state.psi.toFixed(2);
}

function startWorker() {
  if (state.worker) return;
  const worker = new Worker('./kuramoto-worker.js');
  worker.onmessage = (e) => {
    const { type, r, psi } = e.data || {};
    if (type === 'ready') {
      // tick loop at ~60 Hz; crank as needed
      const loop = () => {
        if (!state.running) return;
        worker.postMessage({ type: 'tick' });
        setTimeout(loop, 16);
      };
      loop();
    } else if (type === 'state') {
      state.r = r; state.psi = psi; updateMetrics();
    }
  };
  worker.postMessage({ type: 'init', oscillatorCount: 32, couplingStrength: state.K, dt: 0.01 });
  state.worker = worker;
}

function setBpm(bpm) {
  state.bpm = bpm;
  if (state.workletNode) state.workletNode.port.postMessage({ type: 'set-bpm', bpm });
}

function setCoupling(K) {
  state.K = K;
  if (state.worker) state.worker.postMessage({ type: 'set-k', value: K });
}

async function start() {
  await initAudio();
  startWorker();
  setBpm(parseFloat(inputBpm.value || '120'));
  setCoupling(parseFloat(inputK.value || '1.0'));
  state.running = true;
  btnStart.disabled = true;
  btnStop.disabled = false;
  processVisualQueue();
}

function stop() {
  state.running = false;
  btnStart.disabled = false;
  btnStop.disabled = true;
}

btnStart.addEventListener('click', start);
btnStop.addEventListener('click', stop);
inputBpm.addEventListener('change', (e) => setBpm(parseFloat(e.target.value)));
inputK.addEventListener('change', (e) => setCoupling(parseFloat(e.target.value)));

btnHRV.addEventListener('click', async () => {
  if (!navigator.bluetooth) {
    alert('WebBluetooth required (Chrome, HTTPS).');
    return;
  }
  // Minimal HR monitor connect; full HRV parsing belongs in app-specific code
  try {
    const device = await navigator.bluetooth.requestDevice({ filters: [{ services: ['heart_rate'] }] });
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService('heart_rate');
    const ch = await service.getCharacteristic('heart_rate_measurement');
    await ch.startNotifications();
    ch.addEventListener('characteristicvaluechanged', (event) => {
      const dv = event.target.value;
      const flags = dv.getUint8(0);
      const is16 = (flags & 0x01) !== 0; let off = 1;
      const hr = is16 ? dv.getUint16(off, true) : dv.getUint8(off); off += is16 ? 2 : 1;
      if (flags & 0x08) off += 2; // skip energy expended
      const rr = [];
      if (flags & 0x10) {
        while (off < dv.byteLength) { rr.push((dv.getUint16(off, true) / 1024) * 1000); off += 2; }
      }
      // Map HR to BPM control as a simple demo
      if (hr > 0) setBpm(Math.max(40, Math.min(200, hr)));
    });
    alert('HR monitor connected. BPM follows heart rate.');
  } catch (err) {
    console.warn('HR connect error', err);
    alert('Could not connect HR monitor: ' + err.message);
  }
});

