Hooks Kit (React)

This folder contains an unbundled, TypeScript-first React hooks kit for the WUMBO Neural Engine. It provides:

- useAudioClock: AudioContext-backed master clock with look-ahead scheduling
- useKuramoto: Kuramoto oscillator bank with Heun integration and external-phase injection
- useEntrainment: Composite entrainment controller with adaptive coupling and latency compensation

Usage
- Add to a React project with a bundler (Vite/Next/Rollup). Configure tsconfig to include this folder or copy files into your app.
- Install Zustand for state management.

Example
import { useAudioClock } from './hooks/react/wumbo-hooks';
import { useKuramoto } from './hooks/react/wumbo-hooks';

function Panel(){
  const { currentTime } = useAudioClock();
  const { r, psi, setK, injectExternal } = useKuramoto({ N: 64, K: 1.0 });
  return (
    <div>
      <div>t={currentTime.toFixed(2)} r={r.toFixed(2)} ψ={psi.toFixed(2)}</div>
      <button onClick={()=>setK(1.5)}>K=1.5</button>
      <button onClick={()=>injectExternal(0, Math.PI/2, 0.2)}>Inject</button>
    </div>
  );
}

