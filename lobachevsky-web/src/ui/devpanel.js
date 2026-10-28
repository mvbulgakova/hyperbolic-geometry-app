import {app} from '../app.js';
import {camMatrix} from '../engine/camera.js';
import {canvas} from '../engine/webgl.js';
import {resetCam,fmt} from './hud.js';

// Отладочная панель: клавиша ` (Ё). FPS, разрешение рендера, координаты камеры в модели Кэли — Клейна.
const el=document.createElement('div');
el.id='dev';el.hidden=true;el.className='mono';
el.innerHTML=`<div class="dt">Отладка <span>\`</span></div><div id="devstats"></div>
<label class="check"><input type="checkbox" id="devlock"> зафиксировать масштаб</label>
<label class="ctl">масштаб рендера <output id="devsout"></output><input type="range" id="devscale" min="0.25" max="1" step="0.05" value="0.6"></label>
<button class="btn" id="devreset">камера к старту</button>`;
document.getElementById('ui').appendChild(el);
const st=el.querySelector('#devstats'),lock=el.querySelector('#devlock'),sc=el.querySelector('#devscale');
let frames=0,t0=performance.now(),fps=0;
function tick(now){frames++;if(now-t0>500){fps=frames*1000/(now-t0);frames=0;t0=now;}
  if(lock.checked){app.scale=+sc.value;app.frameAvg=1/50;}
  if(!el.hidden&&app.world){
    const M=camMatrix(),p=[M[12],M[13],M[14],M[15]];
    const k=app.world.type==='flat'?p.slice(0,3):p.slice(0,3).map(x=>x/p[3]);
    st.innerHTML=`<span>мир</span><span>${app.world.id}</span><span>FPS</span><span>${fmt(fps,0)}</span>`+
      `<span>холст</span><span>${canvas.width}×${canvas.height}</span><span>масштаб</span><span>${fmt(app.scale)}</span>`+
      `<span>${app.world.type==='flat'?'x, y, z':'Кэли — Клейн'}</span><span>${k.map(x=>fmt(x,3)).join(' ')}</span>`;
    el.querySelector('#devsout').textContent=fmt(+sc.value);
  }
  requestAnimationFrame(tick);}
requestAnimationFrame(tick);
addEventListener('keydown',e=>{if(e.code==='Backquote'){el.hidden=!el.hidden;}});
el.querySelector('#devreset').addEventListener('click',resetCam);
