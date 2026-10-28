import {I4,O,col,hdist,md,planeThrough,tanDir} from '../engine/math.js';
import {MEADOW} from '../world/meadow.js';
import {fmt} from '../ui/hud.js';
import {app} from '../app.js';

// ───────────────────────── «Пятый постулат»: охрана и обнаружение ─────────────────────────
export const E3=[0,0,1,0], CONE_C=Math.cos(35*Math.PI/180), CONE_R=1.1, R_STAND=0.035, R_CROUCH=0.022;
export const AREF=Math.asin(Math.sinh(R_STAND)/Math.sinh(0.35));
export const perpAt=(X,u)=>planeThrough(X,u,E3);
export function along(X,u,s){const c=Math.cosh(s),h=Math.sinh(s);return {q:X.map((x,i)=>c*x+h*u[i]),f:X.map((x,i)=>h*x+c*u[i])};}
export function mkPatrol(A,B,ph){return {type:'patrol',A,u:tanDir(A,B),L:hdist(A,B),s:ph*hdist(A,B),dir:1,pause:0};}
export function buildStealth(){
  const huts=MEADOW.filter(o=>o.info[0]===2).map(o=>({o,r:hdist(o.q,O)})).sort((a,b)=>b.r-a.r);
  const H=(huts.find(h=>h.r<2.3)||huts[0]).o;
  const goal=along(H.q,tanDir(H.q,O),0.26).q;
  const ug=tanDir(goal,O),pg=perpAt(goal,ug);
  const W=along(goal,pg,-0.3).q;
  const guards=[
    mkPatrol(along(goal,pg,-0.6).q,along(goal,pg,0.6).q,0.0),
    mkPatrol(along(goal,ug,0.5).q,along(goal,ug,1.25).q,0.5),
    {type:'watch',q:W,base:tanDir(W,O),perp:perpAt(W,tanDir(W,O)),ph:0}];
  guards.forEach(updGuard.bind(null,0));
  const yaw=Math.atan2(goal[1],goal[0]);
  return {goal,guards,alert:0,time:0,caught:0,won:false,seen:null,yaw};
}
export function updGuard(dt,g){
  if(g.type==='patrol'){
    if(g.pause>0)g.pause-=dt;
    else{g.s+=g.dir*0.11*dt;
      if(g.s>g.L){g.s=g.L;g.dir=-1;g.pause=1.3;} else if(g.s<0){g.s=0;g.dir=1;g.pause=1.3;}}
    const a=along(g.A,g.u,g.s);g.q=a.q;g.f=a.f.map(x=>x*g.dir);
  }else{g.ph+=dt;const t=1.15*Math.sin(g.ph*0.5);g.f=g.base.map((x,i)=>Math.cos(t)*x+Math.sin(t)*g.perp[i]);}
  g.t2=perpAt(g.q,g.f);
}
export function blocked(Q,u,d,crouch){
  for(const o of MEADOW){const ty=o.info[0];if(ty===4&&!crouch)continue;
    const rad=ty===2?0.16*o.sz:ty===1?0.03*o.sz+0.01:0.07*o.sz;
    if(hdist(Q,o.q)>d+rad+0.01)continue;
    for(let k=1;k<16;k++){if(hdist(along(Q,u,d*k/16).q,o.q)<rad)return true;}}
  return false;
}
export function stealthUpdate(dt){
  const c=app.state[app.world.id],S=c.S;if(!S)return;
  S.time+=dt;S.guards.forEach(updGuard.bind(null,dt));
  const P=col(c.T,3);let best=null;
  if(!S.won)for(const g of S.guards){
    const d=hdist(g.q,P);if(d>CONE_R||d<1e-4)continue;
    const u=tanDir(g.q,P);if(md(u,g.f)<CONE_C)continue;
    if(blocked(g.q,u,d,c.crouch))continue;
    const r=c.crouch?R_CROUCH:R_STAND,a=Math.asin(Math.min(1,Math.sinh(r)/Math.sinh(d)));
    if(!best||a>best.a)best={a,aE:Math.asin(Math.min(1,r/d)),d};
  }
  S.seen=best;
  if(best)S.alert=Math.min(1,S.alert+dt*0.9*best.a/AREF);else S.alert=Math.max(0,S.alert-0.3*dt);
  if(S.alert>=1){S.caught++;S.alert=0;c.T=I4();c.yaw=S.yaw;showMsg('Вас заметили. Начните снова от перекрёстка.',true);}
  if(!S.won&&hdist(P,S.goal)<0.1){S.won=true;S.alert=0;
    showMsg(`Чертежи у вас! Время ${fmt(S.time,0)} с, вас замечали ${S.caught} раз.`,false);
    document.getElementById('sres').textContent=`Пройдено за ${fmt(S.time,0)} с, замечали ${S.caught} раз.`;}
  const bar=document.getElementById('alertfill');bar.style.width=(S.alert*100).toFixed(1)+'%';
  document.getElementById('alertwrap').classList.toggle('hot',!!best);
}
export let msgTimer=0;
export function showMsg(t,bad){const m=document.getElementById('msg');m.textContent=t;m.className=bad?'bad':'good';m.hidden=false;
  clearTimeout(msgTimer);msgTimer=setTimeout(()=>{m.hidden=true;},3200);}
