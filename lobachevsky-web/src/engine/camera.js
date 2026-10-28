import {boostLocal,boostX,boostZ,gs,mul,parabM,rotLocal,rotYP,rotZ} from './math.js';
import {foldPlane,foldRooms,gsFlat,gsPlane} from '../world/geometry.js';
import {GEO,HS,LN2} from '../world/cameras.js';
import {autoStep} from '../world/meadow.js';
import {syncHeight} from '../ui/hud.js';
import {app,inp} from '../app.js';

// ───────────────────────── Движение ─────────────────────────
export function step(dt){
  const c=app.state[app.world.id];
  const run=inp.keys.has('ShiftLeft')||inp.keys.has('ShiftRight')?2.6:1;
  const sp=(app.world.type==='meadow'?0.22:0.55)*run*dt;
  let f=(inp.keys.has('KeyW')||inp.keys.has('ArrowUp')?1:0)-(inp.keys.has('KeyS')||inp.keys.has('ArrowDown')?1:0)+inp.joy[1];
  let s=(inp.keys.has('KeyD')?1:0)-(inp.keys.has('KeyA')?1:0)+inp.joy[0];
  let u=(inp.keys.has('Space')?1:0)-(inp.keys.has('KeyC')||inp.keys.has('ControlLeft')?1:0)+inp.vbtn;
  const turn=(inp.keys.has('ArrowRight')?1:0)-(inp.keys.has('ArrowLeft')?1:0);
  const roll=(inp.keys.has('KeyE')?1:0)-(inp.keys.has('KeyQ')?1:0);
  const sens=0.0026;
  let lx=inp.look[0]*sens+turn*1.6*dt, ly=inp.look[1]*sens; inp.look[0]=0;inp.look[1]=0;
  if(f||s||u||lx||ly||roll) inp.moved=true;
  if(app.world.type==='meadow'&&c.auto){ if(f||s||u){c.auto=null;document.getElementById('result').textContent='Прервано.';} else {autoStep(c,dt);return;} }
  if(app.world.type==='rooms'||app.world.type==='flat'){
    const M=c.M;
    if(lx) rotLocal(M,2,0,lx);
    if(ly) rotLocal(M,2,1,-ly);
    if(roll) rotLocal(M,0,1,roll*1.2*dt);
    if(app.world.type==='rooms'){
      if(f) boostLocal(M,2,f*sp); if(s) boostLocal(M,0,s*sp); if(u) boostLocal(M,1,u*sp);
      gs(M); foldRooms(c,GEO[app.world.id]);
    }else{
      const d=[0,1,2].map(i=>M[8+i]*f*sp+M[i]*s*sp+M[4+i]*u*sp);
      for(let i=0;i<3;i++){c.abs[i]+=d[i];M[12+i]+=d[i];}
      const old=[M[12],M[13],M[14]].map(Math.floor);
      for(let i=0;i<3;i++)M[12+i]-=2*Math.floor(M[12+i]/2);
      const nw=[0,1,2].map(i=>Math.floor(c.abs[i]));
      c.cross=Math.abs(nw[0]-Math.floor(c.start[0]))+Math.abs(nw[1]-Math.floor(c.start[1]))+Math.abs(nw[2]-Math.floor(c.start[2]));
      void old; gsFlat(M);
    }
  }else{
    c.yaw-=lx; c.pitch=Math.max(-1.5,Math.min(1.5,c.pitch-ly));
    const cy=Math.cos(c.yaw),sy=Math.sin(c.yaw);
    const dx=f*cy+s*sy, dy=f*sy-s*cy;
    if(app.world.type==='house'){
      const Z=Math.exp(c.h);
      c.X+=dx*sp*Z*HS; c.Y+=dy*sp*Z*HS;
      c.h+=u*sp;
      while(c.h>=LN2){c.h-=LN2;c.X/=2;c.Y/=2;c.L++;}
      while(c.h<0){c.h+=LN2;c.X*=2;c.Y*=2;c.L--;}
      const W=HS*1048576; if(Math.abs(c.X)>W)c.X-=W*Math.round(c.X/W); if(Math.abs(c.Y)>W)c.Y-=W*Math.round(c.Y/W);
    }else{
      const m=Math.hypot(dx,dy);
      if(m>1e-9){const phi=Math.atan2(dy,dx);c.T=mul(c.T,mul(mul(rotZ(phi),boostX(m*sp*(app.world.stealth&&c.crouch?0.5:1)/Math.cosh(c.h))),rotZ(-phi)));}
      if(app.world.stealth){c.crouch=u<0;c.h=c.crouch?0.04:0.07;}
      else if(u){c.h=Math.max(app.world.type==='meadow'?0.03:0.05,Math.min(3,c.h+u*sp));syncHeight();}
      c.T=gsPlane(c.T); if(app.world.type==='plane')foldPlane(c);
    }
  }
}
export function camMatrix(){
  const c=app.state[app.world.id];
  if(app.world.type==='rooms'||app.world.type==='flat')return c.M;
  if(app.world.type==='house'){const cx=Math.floor(c.X/HS),cy=Math.floor(c.Y/HS);
    return mul(mul(parabM(c.X-cx*HS,c.Y-cy*HS),boostZ(c.h)),rotYP(c.yaw,c.pitch));}
  return mul(mul(c.T,boostZ(c.h)),rotYP(c.yaw,c.pitch));
}
