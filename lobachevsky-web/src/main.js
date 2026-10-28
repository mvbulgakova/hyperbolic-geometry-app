import './engine/input.js';
import './ui/devpanel.js';
import {loadWorlds} from './world/cameras.js';
import {buildNav} from './ui/hud.js';
import {mul,rotZ} from './engine/math.js';
import {U,canvas,fail,gl,loadShader,mapCanvas,mgl,program} from './engine/webgl.js';
import {GEO,HS,PL,WORLDS} from './world/cameras.js';
import {meadowUniforms} from './world/meadow.js';
import {CONE_C,CONE_R,stealthUpdate} from './game/stealth.js';
import {camMatrix,step} from './engine/camera.js';
import {fogEl,glassEl,hud,setWorld} from './ui/hud.js';
import {app} from './app.js';

// ───────────────────────── Кадр ─────────────────────────
export let lastT=performance.now(), hudT=0;
export function resize(){
  const dpr=Math.min(devicePixelRatio||1,2);
  const w=Math.max(160,Math.round(innerWidth*dpr*app.scale)),h=Math.max(120,Math.round(innerHeight*dpr*app.scale));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}
}
export function fovTan(){const asp=innerWidth/innerHeight;return Math.tan((asp<1?82:70)*Math.PI/360);}
export function render(){
  resize();
  gl.viewport(0,0,canvas.width,canvas.height);
  const P=PR[app.world.type]; gl.useProgram(P.pr);
  const c=app.state[app.world.id];
  U(P,'uRes',[canvas.width,canvas.height]); U(P,'uCam',camMatrix()); U(P,'uTanFov',fovTan());
  U(P,'uFog',+fogEl.value); U(P,'uGlass',glassEl.checked?1:0);
  if(app.world.type==='rooms'){const g=GEO[app.world.id];
    U(P,'uM',g.M.flat());U(P,'uE1',g.E1);U(P,'uE2',g.E2);U(P,'uV',g.V);U(P,'uHasV',g.hasV?1:0);
    U(P,'uBall',app.world.p===5?0.085:0.075);U(P,'uTube',0.04);U(P,'uParity',app.world.r%2===0&&app.world.p===5?1:0);U(P,'uCamPar',c.par);}
  else if(app.world.type==='house'){const cx=Math.floor(c.X/HS),cy=Math.floor(c.Y/HS),m=v=>((v%1024)+1024)%1024;
    U(P,'uS',HS);U(P,'uCell',[m(cx),m(cy)]);U(P,'uTube',0.022);}
  else if(app.world.type==='meadow'){const m=meadowUniforms(c),a=2.2,e=0.62;
    U(P,'uQ',m.Q);U(P,'uT1',m.T1);U(P,'uT2',m.T2);U(P,'uI',m.I);U(P,'uNf',m.N);
    U(P,'uDay',app.world.stealth?0.25:1);U(P,'uGQ',m.GQ);U(P,'uGF',m.GF);U(P,'uGn',m.gn);U(P,'uConeC',CONE_C);U(P,'uConeR',CONE_R);
    U(P,'uSun',[Math.cos(e)*Math.cos(a),Math.cos(e)*Math.sin(a),Math.sin(e),1]);}
  else if(app.world.type==='plane'){U(P,'uM',PL.M.flat());U(P,'uE1',PL.E1);U(P,'uE2',PL.E2);U(P,'uCamPar',c.par);U(P,'uLampH',1.1);U(P,'uTube',0.035);}
  else{U(P,'uTube',0.04);U(P,'uBall',0.075);}
  gl.drawArrays(gl.TRIANGLES,0,3);
  if(app.world.type==='plane'&&MAP)renderMap(c);
}
export function renderMap(c){
  const g=mgl,P=MAP;g.viewport(0,0,mapCanvas.width,mapCanvas.height);g.clearColor(0,0,0,0);g.clear(g.COLOR_BUFFER_BIT);g.useProgram(P.pr);
  g.uniform2fv(P.loc.uRes,[mapCanvas.width,mapCanvas.height]);
  g.uniformMatrix4fv(P.loc.uT,false,mul(c.T,rotZ(c.yaw-Math.PI/2)));
  g.uniform4fv(P.loc.uM,PL.M.flat());g.uniform4fv(P.loc.uV,PL.V);g.uniform1f(P.loc.uCamPar,c.par);
  g.uniform1f(P.loc.uFovH,2*Math.atan(fovTan()*innerWidth/innerHeight));
  g.drawArrays(g.TRIANGLES,0,3);
}
export function frame(now){
  const dt=Math.min(0.05,(now-lastT)/1000);lastT=now;
  step(dt);if(app.world.stealth)stealthUpdate(dt);render();
  app.frameAvg=app.frameAvg*0.92+dt*0.08;
  if(app.frameAvg>1/38&&app.scale>0.3){app.scale=Math.max(0.3,app.scale*0.94);app.frameAvg=1/50;}
  else if(app.frameAvg<1/57&&app.scale<1){app.scale=Math.min(1,app.scale*1.03);app.frameAvg=1/50;}
  if(now-hudT>120){hudT=now;hud();}
  requestAnimationFrame(frame);
}


// ───────────────────────── Запуск ─────────────────────────
export const PR={}; export let MAP=null;
try{
  await loadWorlds(); buildNav();
  const VS=await loadShader('fullscreen.vert');
  const [r,h,p,me,f,mp]=await Promise.all(['world_rooms','world_house','world_plane','world_meadow','world_flat','minimap'].map(n=>loadShader(n+'.glsl')));
  PR.rooms=program(gl,r,VS); PR.house=program(gl,h,VS); PR.plane=program(gl,p,VS); PR.meadow=program(gl,me,VS); PR.flat=program(gl,f,VS);
  gl.bindVertexArray(gl.createVertexArray());
  if(mgl){try{MAP=program(mgl,mp,VS);mgl.bindVertexArray(mgl.createVertexArray());}catch(e){MAP=null;}}
}catch(e){fail('Не удалось собрать шейдер:<br><pre style="text-align:left;white-space:pre-wrap;font-size:12px">'+String(e.message).replace(/</g,'&lt;')+'</pre>');throw e;}
document.getElementById('mapwrap').dataset.ok=MAP?'1':'';
export let startId='meadow';try{startId=localStorage.getItem('lob-world2')||startId;}catch(_){}
if(location.hash){const h=location.hash.slice(1);if(WORLDS.some(w=>w.id===h))startId=h;}
setWorld(WORLDS.find(w=>w.id===startId)||WORLDS[0]);
requestAnimationFrame(frame);
