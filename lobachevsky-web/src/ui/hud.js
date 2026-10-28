import {O,apply,col,hdist,md,mul} from '../engine/math.js';
import {houseAbs} from '../world/geometry.js';
import {WORLDS,initCam} from '../world/cameras.js';
import {MAXFLAGS,addFlag,clearFlags,startSquare,triInfo} from '../world/meadow.js';
import {app,inp} from '../app.js';

export const fogEl=document.getElementById('fog'),glassEl=document.getElementById('glass'),hEl=document.getElementById('height');
// ───────────────────────── Интерфейс ─────────────────────────
export const fmt=(x,n=2)=>x.toFixed(n).replace('.',',').replace('-','−');
export const statsEl=document.getElementById('stats');
export function hud(){
  const c=app.state[app.world.id];let rows=[];
  if(app.world.type==='rooms'){const a=apply(c.G,col(c.M,3));rows=[['до начала, ρ',fmt(Math.acosh(Math.max(1,-md(a,c.start))))],['переходов через стены',String(c.cross)]];}
  else if(app.world.type==='flat'){const d=Math.hypot(...c.abs.map((x,i)=>x-c.start[i]));rows=[['до начала',fmt(d)],['кубов от начала',String(c.cross)]];}
  else if(app.world.type==='house'){const a=houseAbs(c),s=c.start,d2=(a[0]-s[0])**2+(a[1]-s[1])**2+(a[2]-s[2])**2;
    rows=[['до начала, ρ',fmt(Math.acosh(1+d2/(2*a[2]*s[2])))],['этаж',(c.L>0?'+':c.L<0?'−':'')+Math.abs(c.L)],['высота над полом',fmt(c.h)]];}
  else if(app.world.stealth){const S=c.S,a=col(c.T,3);
    rows=[['до сундука',S.won?'взят':fmt(hdist(a,S.goal))],['тревога',Math.round(S.alert*100)+'%'],['поза',c.crouch?'пригнулись':'стоите']];
    if(S.seen)rows.push(['вас видят, расстояние',fmt(S.seen.d)],['угловой радиус α',fmt(S.seen.a*180/Math.PI,2)+'°'],['в евклидовом мире',fmt(S.seen.aE*180/Math.PI,2)+'°']);
    else rows.push(['вас видят','нет']);}
  else if(app.world.type==='meadow'){const a=col(c.T,3),P=2*Math.atan(Math.exp(-c.h));
    rows=[['до перекрёстка, ρ',fmt(hdist(a,O))],['высота глаз',fmt(c.h)],['флажков',c.flags.length+' из '+MAXFLAGS],['Π(h)',fmt(P*180/Math.PI,1)+'°']];
    if(c.flags.length>=3){const t=triInfo(c.flags);
      rows.push(['углы',[t.a,t.b,t.c].map(x=>fmt(x*180/Math.PI,1)+'°').join(' + ')],['сумма',fmt(t.sum*180/Math.PI,1)+'°'],['площадь S = π − сумма',fmt(t.defect,3)+' k²'],['стороны',t.sides.map(x=>fmt(x)).join(' · ')]);}}
  else{const a=apply(mul(c.G,c.T),O),P=2*Math.atan(Math.exp(-c.h));
    rows=[['до начала по полу',fmt(Math.acosh(Math.max(1,-md(a,c.start))))],['высота h',fmt(c.h)],['Π(h)',fmt(P*180/Math.PI,1)+'°'],['край пола ниже горизонта на',fmt(90-P*180/Math.PI,1)+'°']];}
  statsEl.innerHTML=rows.map(([k,v])=>`<span class="k">${k}</span><span class="v">${v}</span>`).join('');
}
export const worldsEl=document.getElementById('worlds');
export function buildNav(){
WORLDS.forEach((w,i)=>{const b=document.createElement('button');b.innerHTML=`<kbd>${i+1}</kbd>${w.short||w.name}`;
  b.dataset.id=w.id;b.addEventListener('click',()=>setWorld(w));worldsEl.appendChild(b);});
}
export function setWorld(w){
  app.world=w; if(!app.state[w.id])app.state[w.id]=initCam(w);
  document.getElementById('wname').textContent=w.name;
  document.getElementById('wsub').textContent=w.sub;
  document.getElementById('wdesc').textContent=w.desc;
  document.getElementById('wformula').innerHTML=w.formula;
  document.getElementById('wtasks').innerHTML=w.tasks.map(t=>`<li>${t}</li>`).join('');
  for(const b of worldsEl.children)b.setAttribute('aria-pressed',b.dataset.id===w.id);
  document.getElementById('mapwrap').hidden=w.type!=='plane'||!document.getElementById('mapwrap').dataset.ok;
  document.getElementById('hrow').hidden=!(w.type==='plane'||w.type==='meadow');
  document.getElementById('mtools').hidden=w.type!=='meadow'||!!w.stealth;
  document.getElementById('stools').hidden=!w.stealth;
  document.getElementById('alertwrap').hidden=!w.stealth;
  if(w.stealth)document.getElementById('hrow').hidden=true;
  hEl.min=w.type==='meadow'?0.03:0.05;
  fogEl.value=w.fog;syncFog();syncHeight();
  try{localStorage.setItem('lob-world2',w.id);}catch(_){}
  hud();
}
export function resetCam(){app.state[app.world.id]=initCam(app.world);syncHeight();}
export function syncFog(){document.getElementById('fout').textContent=fmt(+fogEl.value,1);}
export function syncHeight(){if(app.world.type!=='plane'&&app.world.type!=='meadow')return;const c=app.state[app.world.id];hEl.value=c.h;document.getElementById('hout').textContent=fmt(c.h);}
fogEl.addEventListener('input',syncFog);
hEl.addEventListener('input',()=>{if(app.world.type==='plane'||app.world.type==='meadow'){app.state[app.world.id].h=+hEl.value;syncHeight();}});
document.getElementById('reset').addEventListener('click',resetCam);
document.getElementById('bflagp').addEventListener('click',()=>{if(app.world.type==='meadow')addFlag(app.state[app.world.id]);});
document.getElementById('bflag').addEventListener('click',()=>{if(app.world.type==='meadow')addFlag(app.state[app.world.id]);});
document.getElementById('bclear').addEventListener('click',clearFlags);
document.getElementById('bsquare').addEventListener('click',startSquare);
document.getElementById('srestart').addEventListener('click',()=>{resetCam();document.getElementById('sres').textContent='';});
document.getElementById('side').addEventListener('input',e=>{document.getElementById('sout').textContent=fmt(+e.target.value);});
document.getElementById('fs').addEventListener('click',()=>{try{const r=document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();if(r&&r.catch)r.catch(()=>{});}catch(_){}});
export const panel=document.getElementById('panel'),pt=document.getElementById('ptoggle');
export function togglePanel(){panel.classList.toggle('closed');const open=!panel.classList.contains('closed');pt.setAttribute('aria-expanded',open);pt.querySelector('span').textContent=open?'−':'+';}
pt.addEventListener('click',togglePanel);
export const coarse=matchMedia('(pointer:coarse)').matches;
if(coarse||innerWidth<760)togglePanel();
export const hint=document.getElementById('hint');
hint.textContent=coarse?'Левый круг — идти, ведите пальцем по экрану — смотреть, ▲ ▼ — вверх и вниз'
  :'Щёлкните по сцене и смотрите мышью · W A S D — идти · Пробел и C — вверх и вниз · Shift — быстрее · F — флажок · Q и E — наклон · R — к началу · 1–8 — миры';
setInterval(()=>{if(inp.moved)hint.classList.add('gone');},4000);
