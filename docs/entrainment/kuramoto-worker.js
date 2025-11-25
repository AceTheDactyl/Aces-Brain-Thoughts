let phases, frequencies, N, K, dt;
function gaussianRandom(){ const u1=Math.random(), u2=Math.random(); return Math.sqrt(-2*Math.log(u1))*Math.cos(2*Math.PI*u2); }
function init(count,coupling,step){ N=count; K=coupling; dt=step; phases=new Float32Array(N); frequencies=new Float32Array(N);
  for(let i=0;i<N;i++){ phases[i]=Math.random()*2*Math.PI; frequencies[i]=1.0+gaussianRandom()*0.2; } }
function orderParameter(){ let r=0, i_=0; for(let k=0;k<N;k++){ r+=Math.cos(phases[k]); i_+=Math.sin(phases[k]); } r/=N; i_/=N; const mag=Math.hypot(r,i_); const psi=Math.atan2(i_,r); return { r:mag, psi }; }
function step(){ const {r,psi}=orderParameter(); const next=new Float32Array(N);
  for(let k=0;k<N;k++){ const d1=frequencies[k]+K*r*Math.sin(psi-phases[k]); const pred=phases[k]+dt*d1; const d2=frequencies[k]+K*r*Math.sin(psi-pred); let p=phases[k]+dt*(d1+d2)/2; p%=2*Math.PI; if(p<0) p+=2*Math.PI; next[k]=p; }
  phases=next; return { r, psi }; }
self.onmessage=(e)=>{ const {type}=e.data||{}; if(type==='init'){ const {oscillatorCount,couplingStrength=1.0,dt:step=0.01}=e.data; init(oscillatorCount,couplingStrength,step); self.postMessage({type:'ready'}); }
  else if(type==='tick'){ const {r,psi}=step(); self.postMessage({type:'state', r, psi}); }
  else if(type==='set-k'){ K=e.data.value; } };

