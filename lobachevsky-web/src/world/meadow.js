import {boostX,col,hdist,md,mul,rotZ,tanDir} from '../engine/math.js';
import {gsPlane} from './geometry.js';
import {fmt} from '../ui/hud.js';
import {app,inp} from '../app.js';

// ───────────────────────── Луг: сцена и инструменты ─────────────────────────
export function buildMeadow(){
  let seed=7;const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;};
  const objs=[];
  const add=(type,sz,R,rho,phi)=>{
    const sh=Math.sinh(rho),ch=Math.cosh(rho),cp=Math.cos(phi),sp=Math.sin(phi);
    const q=[sh*cp,sh*sp,0,ch];
    if(Math.min(Math.abs(Math.asinh(q[0])),Math.abs(Math.asinh(q[1])))<0.14+0.12*sz)return false;
    for(const o of objs)if(Math.acosh(Math.max(1,-md(q,o.q)))<0.34*(sz+o.sz))return false;
    objs.push({q,t1:[ch*cp,ch*sp,0,sh],t2:[-sp,cp,0,0],info:[type,sz,0,R],sz});return true;
  };
  const place=(n,type,szLo,szHi,R,rLo,rHi)=>{let k=0,tries=0;
    while(k<n&&tries++<400){const sz=szLo+(szHi-szLo)*rnd();if(add(type,sz,R*sz,rLo+(rHi-rLo)*rnd(),rnd()*2*Math.PI))k++;}};
  place(4,2,0.9,1.2,0.40,0.9,2.2);
  place(17,1,0.8,1.3,0.50,0.5,3.3);
  place(5,4,0.8,1.6,0.12,0.4,2.6);
  return objs;
}
export const MEADOW=buildMeadow();
export const MAXFLAGS=12;
export function addFlag(c,fwd=0.15,side=0){
  if(c.flags.length>=MAXFLAGS)return false;
  let F=mul(c.T,mul(rotZ(c.yaw),boostX(fwd)));
  if(side)F=mul(F,mul(rotZ(Math.PI/2),boostX(side)));
  F=gsPlane(F);
  c.flags.push({q:col(F,3).slice(),t1:col(F,0).slice(),t2:col(F,1).slice()});return true;
}
export function triInfo(flags){
  const [A,B,C]=flags.slice(-3).map(o=>o.q);
  const ang=(P,Q,R)=>Math.acos(Math.max(-1,Math.min(1,md(tanDir(P,Q),tanDir(P,R)))));
  const a=ang(A,B,C),b=ang(B,A,C),c=ang(C,A,B);
  return {a,b,c,sum:a+b+c,defect:Math.PI-(a+b+c),sides:[hdist(B,C),hdist(A,C),hdist(A,B)]};
}
export function meadowUniforms(c){
  const N0=0,Q=new Float32Array(160),T1=new Float32Array(160),T2=new Float32Array(160),I=new Float32Array(160);
  const all=c.S?MEADOW.concat(c.S.guards.map(g=>({q:g.q,t1:g.f,t2:g.t2,info:[5,1,0,0.14]})),c.S.won?[]:[{q:c.S.goal,t1:[1,0,0,0],t2:[0,1,0,0],info:[6,1,0,0.1]}])
    :MEADOW.concat(c.flags.map(f=>({q:f.q,t1:f.t1,t2:f.t2,info:[3,1,0,0.30]})));
  all.forEach((o,i)=>{Q.set(o.q,i*4);T1.set(o.t1,i*4);T2.set(o.t2,i*4);I.set(o.info,i*4);});
  const GQ=new Float32Array(16),GF=new Float32Array(16);let gn=0;
  if(c.S)c.S.guards.forEach((g,i)=>{GQ.set(g.q,i*4);GF.set(g.f,i*4);gn++;});
  return {N:all.length+N0,Q,T1,T2,I,GQ,GF,gn};
}
export function startSquare(){
  const c=app.state[app.world.id];if(!c)return;
  const L=+document.getElementById('side').value;
  c.flags=[];addFlag(c,0,0.10);c.auto={leg:0,rem:L,L,turn:0,start:col(c.T,3).slice()};
  document.getElementById('result').textContent='Иду…';inp.moved=true;
}
export function autoStep(c,dt){
  const a=c.auto;
  if(a.turn>0){const d=Math.min(a.turn,2.4*dt);c.yaw-=d;a.turn-=d;return;}
  const ds=Math.min(a.rem,0.45*dt);
  const phi=c.yaw;c.T=gsPlane(mul(c.T,mul(mul(rotZ(phi),boostX(ds)),rotZ(-phi))));
  a.rem-=ds;
  if(a.rem<=1e-9){
    addFlag(c,0,0.10);a.leg++;
    if(a.leg>=4){
      const gap=hdist(col(c.T,3),a.start);
      document.getElementById('result').innerHTML=`Четыре стороны по ${fmt(a.L)} и четыре поворота на 90°. До начала осталось ${fmt(gap)}. На евклидовой плоскости было бы 0.`;
      c.auto=null;
    }else{a.turn=Math.PI/2;a.rem=a.L;}
  }
}
export function clearFlags(){const c=app.state[app.world.id];if(!c)return;c.flags=[];c.auto=null;document.getElementById('result').textContent='';}
