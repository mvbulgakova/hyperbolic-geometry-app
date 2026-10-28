

// ───────────────────────── WebGL ─────────────────────────
export const canvas=document.getElementById('gl');
export const gl=canvas.getContext('webgl2',{antialias:false,powerPreference:'high-performance'});
export function fail(msg){const d=document.createElement('div');d.id='err';d.innerHTML=msg;document.body.appendChild(d);}
if(!gl){fail('Этому браузеру не хватает WebGL2.<br>Откройте страницу в свежем Chrome, Firefox, Edge или Safari.');throw new Error('no webgl2');}
export function program(g,fs,VS){
  const mk=(t,s)=>{const sh=g.createShader(t);g.shaderSource(sh,s);g.compileShader(sh);
    if(!g.getShaderParameter(sh,g.COMPILE_STATUS))throw new Error(g.getShaderInfoLog(sh));return sh;};
  const pr=g.createProgram();g.attachShader(pr,mk(g.VERTEX_SHADER,VS));g.attachShader(pr,mk(g.FRAGMENT_SHADER,fs));g.linkProgram(pr);
  if(!g.getProgramParameter(pr,g.LINK_STATUS))throw new Error(g.getProgramInfoLog(pr));
  const loc={};const n=g.getProgramParameter(pr,g.ACTIVE_UNIFORMS);
  for(let i=0;i<n;i++){const u=g.getActiveUniform(pr,i);const nm=u.name.replace(/\[0\]$/,'');loc[nm]=g.getUniformLocation(pr,u.name);}
  return {pr,loc};
}


export const mapCanvas=document.getElementById('map');
export const mgl=mapCanvas.getContext('webgl2',{premultipliedAlpha:true,alpha:true});


export function U(P,name,val){const l=P.loc[name];if(l==null)return;
  if(typeof val==='number')gl.uniform1f(l,val);
  else if(val.length===2)gl.uniform2fv(l,val);
  else if(name==='uCam')gl.uniformMatrix4fv(l,false,val);
  else gl.uniform4fv(l,val.flat?val.flat():val);}
// Шейдеры лежат в src/shaders/*.glsl; строка #include "файл" подставляет другой файл.
export const SH=new URL('../shaders/',import.meta.url);
export const cache={};
export async function loadShader(name){
  if(!cache[name])cache[name]=fetch(new URL(name,SH)).then(r=>{if(!r.ok)throw new Error('не загрузился шейдер '+name);return r.text();});
  const t=await cache[name];
  const inc=[...t.matchAll(/#include "([^"]+)"/g)].map(m=>m[1]);
  let out=t;for(const n of inc)out=out.replace(`#include "${n}"`,(await loadShader(n)).replace(/\n$/,''));
  return out;
}
