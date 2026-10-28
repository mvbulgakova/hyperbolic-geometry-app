import {canvas} from './webgl.js';
import {WORLDS} from '../world/cameras.js';
import {addFlag,clearFlags} from '../world/meadow.js';
import {resetCam,setWorld,togglePanel} from '../ui/hud.js';
import {app,inp} from '../app.js';

// ───────────────────────── Ввод ─────────────────────────
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'&&e.target.type==='range')return;
  const k=e.code;
  if(/^Digit[1-8]$/.test(k)){setWorld(WORLDS[+k.slice(5)-1]);return;}
  if(k==='KeyR'){resetCam();return;}
  if(k==='KeyF'&&app.world.type==='meadow'&&!app.world.stealth){addFlag(app.state[app.world.id]);return;}
  if(k==='KeyX'&&app.world.type==='meadow'&&!app.world.stealth){clearFlags();return;}
  if(k==='KeyH'){togglePanel();return;}
  inp.keys.add(k); if(['Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(k))e.preventDefault();
});
addEventListener('keyup',e=>inp.keys.delete(e.code));
addEventListener('blur',()=>inp.keys.clear());
export let dragging=false,last=[0,0];
canvas.addEventListener('mousedown',e=>{dragging=true;last=[e.clientX,e.clientY];
  if(!document.pointerLockElement&&canvas.requestPointerLock){try{const r=canvas.requestPointerLock();if(r&&r.catch)r.catch(()=>{});}catch(_){}}});
addEventListener('mouseup',()=>dragging=false);
addEventListener('mousemove',e=>{
  if(document.pointerLockElement===canvas){inp.look[0]+=e.movementX;inp.look[1]+=e.movementY;}
  else if(dragging){inp.look[0]+=e.clientX-last[0];inp.look[1]+=e.clientY-last[1];last=[e.clientX,e.clientY];}
});
export const touches=new Map();
canvas.addEventListener('touchstart',e=>{for(const t of e.changedTouches)touches.set(t.identifier,[t.clientX,t.clientY]);e.preventDefault();},{passive:false});
canvas.addEventListener('touchmove',e=>{for(const t of e.changedTouches){const p=touches.get(t.identifier);if(p){inp.look[0]+=(t.clientX-p[0])*1.4;inp.look[1]+=(t.clientY-p[1])*1.4;touches.set(t.identifier,[t.clientX,t.clientY]);}}e.preventDefault();},{passive:false});
canvas.addEventListener('touchend',e=>{for(const t of e.changedTouches)touches.delete(t.identifier);});
export const joyEl=document.getElementById('joy'),knob=document.getElementById('knob');let joyId=null;
export function joyMove(t){const r=joyEl.getBoundingClientRect();let x=t.clientX-r.left-60,y=t.clientY-r.top-60;const l=Math.hypot(x,y);if(l>48){x*=48/l;y*=48/l;}
  knob.style.transform=`translate(${x}px,${y}px)`;inp.joy=[x/48,-y/48];inp.moved=true;}
joyEl.addEventListener('touchstart',e=>{const t=e.changedTouches[0];joyId=t.identifier;joyMove(t);e.preventDefault();},{passive:false});
joyEl.addEventListener('touchmove',e=>{for(const t of e.changedTouches)if(t.identifier===joyId)joyMove(t);e.preventDefault();},{passive:false});
export const joyEnd=e=>{for(const t of e.changedTouches)if(t.identifier===joyId){joyId=null;inp.joy=[0,0];knob.style.transform='';}};
joyEl.addEventListener('touchend',joyEnd);joyEl.addEventListener('touchcancel',joyEnd);
for(const [id,dir] of [['bup',1],['bdown',-1]]){const b=document.getElementById(id);
  b.addEventListener('pointerdown',e=>{inp.vbtn=dir;b.setPointerCapture(e.pointerId);});
  b.addEventListener('pointerup',()=>inp.vbtn=0);b.addEventListener('pointercancel',()=>inp.vbtn=0);}
