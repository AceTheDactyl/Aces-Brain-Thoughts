class TimingWorkletProcessor extends AudioWorkletProcessor {
  constructor(options){ super(options); this._nextBeatFrame=0; this._framesPerBeat=Math.floor(sampleRate*60/120);
    this.port.onmessage=(e)=>{ const {type,bpm}=e.data||{}; if(type==='set-bpm' && typeof bpm==='number'){ this._framesPerBeat=Math.max(32, Math.floor(sampleRate*60/bpm)); } };
  }
  static get parameterDescriptors(){ return [{ name:'tempo', defaultValue:120, minValue:20, maxValue:300, automationRate:'k-rate' }]; }
  process(_i,_o,parameters){ const tempo=parameters.tempo?.[0]??120; const fpb=Math.max(32, Math.floor(sampleRate*60/tempo)); if(fpb!==this._framesPerBeat) this._framesPerBeat=fpb;
    for(let i=0;i<128;i++){ const abs=currentFrame+i; if(abs>=this._nextBeatFrame){ this._nextBeatFrame=abs+this._framesPerBeat; this.port.postMessage({type:'beat', frame:abs, time:abs/sampleRate, audioContextTime:currentTime}); } }
    return true; }
}
registerProcessor('timing-processor', TimingWorkletProcessor);

