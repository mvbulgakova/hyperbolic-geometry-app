import {O,col,foot,gs,md,mul,orient,orth,planeThrough,refl,setCol} from '../engine/math.js';
import {PL} from './cameras.js';

export function roomGeom(p,r){
  let us,c;
  if(p===4){us=[[1,0,0],[0,1,0],[0,0,1]];c=0;}
  else{const f=(1+Math.sqrt(5))/2,n=Math.hypot(1,f);us=[[0,1,f],[1,f,0],[f,0,1]].map(u=>u.map(x=>x/n));c=1/Math.sqrt(5);}
  const th=2*Math.PI/r,ch2=(1+Math.cos(th))/(1-c),chd=Math.sqrt(ch2),shd=Math.sqrt(ch2-1);
  const N=us.map(u=>[chd*u[0],chd*u[1],chd*u[2],shd]);
  const P=[O,foot([N[0]]),foot([N[0],N[1]]),foot(N)];
  const M=[0,1,2,3].map(k=>{const o=P.filter((_,i)=>i!==k);return orient(planeThrough(...o),P[k]);});
  const ideal=md(P[3],P[3])>-1e-9;
  return {M,E1:N[0],E2:orth(N[1],N[0]),V:ideal?[0,0,0,1]:P[3],hasV:!ideal,t:shd/chd,R:M.map(refl)};
}
export function planeGeom(){
  const c=Math.cos(2*Math.PI/5),ch2=1/(1-c),chd=Math.sqrt(ch2),shd=Math.sqrt(ch2-1);
  const a=2*Math.PI/5;
  const N0=[chd,0,0,shd],N1=[chd*Math.cos(a),chd*Math.sin(a),0,shd];
  const Mi=foot([N0]),V=foot([N0,N1]),e3=[0,0,1,0];
  const M=[orient(planeThrough(Mi,V,e3),O),orient(planeThrough(O,V,e3),Mi),orient(planeThrough(O,Mi,e3),V)];
  return {M,E1:N0,E2:orth(N1,N0),V,R:M.map(refl),delta:Math.acosh(chd)};
}
export function foldRooms(c,g){
  for(let it=0;it<24;it++){let any=false;
    for(let k=0;k<4;k++){if(md(col(c.M,3),g.M[k])>0){c.M=mul(g.R[k],c.M);c.G=mul(c.G,g.R[k]);if(k===0){c.cross++;c.par^=1;}any=true;}}
    if(!any)break;}
}
export function foldPlane(c){
  for(let it=0;it<24;it++){let any=false;
    for(let k=0;k<3;k++){if(md(col(c.T,3),PL.M[k])>0){c.T=mul(PL.R[k],c.T);c.G=mul(c.G,PL.R[k]);if(k===0)c.par^=1;any=true;}}
    if(!any)break;}
}
export function gsPlane(T){
  const t=T.slice();for(let c=0;c<4;c++)t[c*4+2]=0;setCol(t,2,[0,0,1,0]);
  gs(t);setCol(t,2,[0,0,1,0]);for(let c=0;c<4;c++)if(c!==2)t[c*4+2]=0;return t;
}
export function houseAbs(c){const f=Math.pow(2,c.L);return [c.X*f,c.Y*f,Math.exp(c.h)*f];}
export function gsFlat(M){const v=[0,1,2].map(i=>[M[i*4],M[i*4+1],M[i*4+2]]);
  const nrm=a=>{const l=Math.hypot(...a);return a.map(x=>x/l)},dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  v[0]=nrm(v[0]);v[1]=nrm(v[1].map((x,i)=>x-dot(v[1],v[0])*v[0][i]));
  v[2]=nrm(v[2].map((x,i)=>x-dot(v[2],v[0])*v[0][i]-dot(v[2],v[1])*v[1][i]));
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)M[i*4+j]=v[i][j];}
