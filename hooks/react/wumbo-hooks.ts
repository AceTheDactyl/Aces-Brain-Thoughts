/*
  WUMBO Hooks Kit (React/TypeScript, unbundled)
  - useAudioClock: AudioContext-backed master clock
  - useKuramoto: Kuramoto oscillator bank with Heun integration
  - useEntrainment: Adaptive coupling + latency compensation wrapper
*/
import { useEffect, useRef, useState, useMemo } from 'react';
import create from 'zustand';

export function useAudioClock() {
  const [ctx, setCtx] = useState<AudioContext | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const ac = new (window.AudioContext || (window as any).webkitAudioContext)();
    setCtx(ac);
    let running = true;
    const tick = () => { if (!running) return; setCurrentTime(ac.currentTime); rafRef.current = requestAnimationFrame(tick); };
    rafRef.current = requestAnimationFrame(tick);
    return () => { running = false; if (rafRef.current) cancelAnimationFrame(rafRef.current); ac.close(); };
  }, []);

  return { audioContext: ctx, currentTime };
}

export type KuramotoOptions = { N?: number; K?: number; dt?: number; sigma?: number; noise?: number };

function randn() { const u1 = Math.random(), u2 = Math.random(); return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }

export function useKuramoto(opts: KuramotoOptions = {}) {
  const TAU = Math.PI * 2;
  const { N, K, dt, sigma, noise } = { N: 64, K: 1.0, dt: 0.01, sigma: 0.2, noise: 0.0, ...opts };
  const phasesRef = useRef<Float32Array>(new Float32Array(N).map(() => Math.random() * TAU));
  const freqRef = useRef<Float32Array>(new Float32Array(N).map(() => 1.0 + randn() * sigma));
  const [state, setState] = useState({ r: 0, psi: 0, K });

  const setK = (k: number) => setState(s => ({ ...s, K: k }));
  const shuffle = () => { const f = freqRef.current; for (let i = 0; i < f.length; i++) f[i] = 1.0 + randn() * sigma; };
  const reset = () => { phasesRef.current = new Float32Array(N).map(() => Math.random() * TAU); };
  const order = () => {
    let sx = 0, sy = 0; const p = phasesRef.current; for (let i = 0; i < p.length; i++) { sx += Math.cos(p[i]); sy += Math.sin(p[i]); }
    sx /= p.length; sy /= p.length; return { r: Math.hypot(sx, sy), psi: Math.atan2(sy, sx) };
  };
  const step = (externalPhase?: number, injectStrength: number = 0) => {
    const p = phasesRef.current; const f = freqRef.current; const { r, psi } = order();
    const np = new Float32Array(p.length);
    for (let i = 0; i < p.length; i++) {
      const inj = externalPhase == null ? 0 : injectStrength * Math.sin(externalPhase - p[i]);
      const noiseTerm = noise * randn();
      const dtheta1 = f[i] + state.K * r * Math.sin(psi - p[i]) + inj + noiseTerm;
      const pred = p[i] + dt * dtheta1;
      const dtheta2 = f[i] + state.K * r * Math.sin(psi - pred) + inj + noiseTerm;
      let th = p[i] + dt * (dtheta1 + dtheta2) / 2; th %= TAU; if (th < 0) th += TAU; np[i] = th;
    }
    phasesRef.current = np; setState(s => ({ ...s, r, psi }));
  };

  const injectExternal = (index: number, phase: number, strength: number) => {
    const p = phasesRef.current; const diff = phase - p[index]; p[index] += strength * Math.sin(diff); p[index] = ((p[index] % (TAU)) + TAU) % TAU;
  };

  return { r: state.r, psi: state.psi, setK, shuffle, reset, step, injectExternal, phases: phasesRef };
}

type EntrainmentState = { couplingStrength: number; entrainmentScore: number };

export function useEntrainment(kuramoto: ReturnType<typeof useKuramoto>) {
  const [state, setState] = useState<EntrainmentState>({ couplingStrength: 1.0, entrainmentScore: 0 });
  const historyRef = useRef<number[]>([]);

  const process = (userPhase: number) => {
    // Step Kuramoto, compute phase diff
    kuramoto.step();
    const phaseDiff = (() => {
      let d = userPhase - kuramoto.psi; if (d > Math.PI) d -= 2 * Math.PI; if (d < -Math.PI) d += 2 * Math.PI; return d;
    })();
    const entrainmentScore = Math.cos(phaseDiff) * 0.5 + 0.5;
    historyRef.current.push(entrainmentScore); if (historyRef.current.length > 100) historyRef.current.shift();
    const avg = historyRef.current.reduce((a, b) => a + b, 0) / historyRef.current.length;
    // Adaptive K
    const target = 0.7; const rate = 0.01; const minK = 0.1, maxK = 5.0;
    let newK = kuramoto.r + rate * (target - avg);
    newK = Math.min(maxK, Math.max(minK, newK)); kuramoto.setK(newK);
    setState({ couplingStrength: newK, entrainmentScore: avg });
  };

  const compensateLatency = (phase: number, frequency: number, latencyMs: number) => {
    const adv = (frequency * latencyMs / 1000) * Math.PI * 2; let p = phase + adv; p %= Math.PI * 2; if (p < 0) p += Math.PI * 2; return p;
  };

  return { ...state, process, compensateLatency };
}

