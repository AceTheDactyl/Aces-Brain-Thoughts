/* Kuramoto solver in a Worker. Supports SharedArrayBuffer if cross-origin isolated. */
let phases, frequencies, state, N, K, dt;

function gaussianRandom() {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function initLocal(count, coupling, step) {
  N = count; K = coupling; dt = step;
  phases = new Float32Array(N);
  frequencies = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    phases[i] = Math.random() * 2 * Math.PI;
    frequencies[i] = 1.0 + gaussianRandom() * 0.2;
  }
}

function orderParameter() {
  let sumReal = 0, sumImag = 0;
  for (let i = 0; i < N; i++) {
    sumReal += Math.cos(phases[i]);
    sumImag += Math.sin(phases[i]);
  }
  sumReal /= N; sumImag /= N;
  const r = Math.hypot(sumReal, sumImag);
  const psi = Math.atan2(sumImag, sumReal);
  return { r, psi };
}

function step() {
  const { r, psi } = orderParameter();
  const newPhases = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const dtheta1 = frequencies[i] + K * r * Math.sin(psi - phases[i]);
    const thetaPred = phases[i] + dt * dtheta1;
    const dtheta2 = frequencies[i] + K * r * Math.sin(psi - thetaPred);
    let next = phases[i] + dt * (dtheta1 + dtheta2) / 2;
    next %= 2 * Math.PI; if (next < 0) next += 2 * Math.PI;
    newPhases[i] = next;
  }
  phases = newPhases;
  return { r, psi };
}

self.onmessage = (e) => {
  const { type } = e.data || {};
  if (type === 'init') {
    const { oscillatorCount, couplingStrength = 1.0, dt: step = 0.01 } = e.data;
    if (e.data.sharedPhases && e.data.sharedState && self.SharedArrayBuffer) {
      // Optional SAB path (not used by default demo to avoid COOP/COEP needs)
      phases = new Float32Array(e.data.sharedPhases);
      state = new Float32Array(e.data.sharedState);
      N = oscillatorCount; K = couplingStrength; dt = step;
      // Initialize phases/frequencies locally for now
      frequencies = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        phases[i] = Math.random() * 2 * Math.PI;
        frequencies[i] = 1.0 + gaussianRandom() * 0.2;
      }
    } else {
      initLocal(oscillatorCount, couplingStrength, step);
    }
    self.postMessage({ type: 'ready' });
  } else if (type === 'tick') {
    const { r, psi } = step();
    self.postMessage({ type: 'state', r, psi });
  } else if (type === 'set-k') {
    K = e.data.value;
  }
};

