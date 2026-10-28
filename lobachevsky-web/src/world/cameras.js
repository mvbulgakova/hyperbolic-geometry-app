import {I4,O,apply,boostLocal,col,mul,setCol} from '../engine/math.js';
import {foldRooms,houseAbs,planeGeom,roomGeom} from './geometry.js';
import {buildStealth} from '../game/stealth.js';

export const WORLDS=[];
// ───────────────────────── Камеры ─────────────────────────
export const GEO={};
export async function loadWorlds(){
  const r=await fetch(new URL('../../data/worlds.json',import.meta.url));
  if(!r.ok)throw new Error('не загрузился data/worlds.json');
  WORLDS.push(...await r.json());
  for(const w of WORLDS) if(w.type==='rooms') GEO[w.id]=roomGeom(w.p,w.r);
}
export const PL=planeGeom();
export const LN2=Math.LN2, HS=1.0;


export function initCam(w){
  if(w.type==='rooms'){
    const M=I4();boostLocal(M,2,-0.6);boostLocal(M,0,0.12);boostLocal(M,1,0.2);
    const c={M,G:I4(),cross:0,par:0};foldRooms(c,GEO[w.id]);c.cross=0;
    c.start=apply(c.G,col(c.M,3));return c;
  }
  if(w.type==='house'){const c={X:0.5*HS,Y:0.18*HS,L:0,h:0.26,yaw:Math.PI/2,pitch:0.12};c.start=houseAbs(c);return c;}
  if(w.type==='meadow'){const c={T:I4(),G:I4(),h:0.07,yaw:0,pitch:0.03,par:0,flags:[],auto:null,crouch:false};c.start=O.slice();
    if(w.stealth){c.S=buildStealth();c.yaw=c.S.yaw;}return c;}
  if(w.type==='plane'){const c={T:I4(),G:I4(),h:0.45,yaw:0.35,pitch:-0.12,par:0};c.start=apply(mul(c.G,c.T),O);return c;}
  const M=I4();setCol(M,3,[0.5,0.5,-0.1,1]);return {M,abs:[0.5,0.5,-0.1],cross:0,start:[0.5,0.5,-0.1]};
}
