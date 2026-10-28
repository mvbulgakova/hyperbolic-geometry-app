#include "base.glsl"

uniform float uS; uniform vec2 uCell;
const float LN2=0.6931471806;
const vec3 FOG=vec3(0.030,0.030,0.058);
vec2 gCell;
void parab(inout vec4 x, float a, float b){ float w=x.w-x.z, u=x.w+x.z; float x1=x.x+a*w, x2=x.y+b*w;
  u+=2.*a*x.x+2.*b*x.y+(a*a+b*b)*w; x=vec4(x1,x2,(u-w)*.5,(u+w)*.5); }
void dil(inout vec4 x, float a){ float c=cosh(a), s=sinh(a); x.zw=vec2(c*x.z+s*x.w, s*x.z+c*x.w); }
vec4 fromHS(vec3 q){ float w=1./q.z; float u=dot(q,q)/q.z; return vec4(q.x*w,q.y*w,(u-w)*.5,(u+w)*.5); }
float map(vec4 p){
  float w=p.w-p.z, Z=1./w, X=p.x*Z, Y=p.y*Z, lz=log(Z);
  vec2 q=vec2(X,Y);
  vec2 c=vec2(X<uS*.5?0.:uS, Y<uS*.5?0.:uS);
  float d=asinh(length(q-c)*w)-uTube*1.25; gId=1;
  float bx=min(X,uS-X), by=min(Y,uS-Y);
  float a1=asinh(min(bx,by)*w);
  float b1=sqrt(lz*lz+a1*a1)-uTube; if(b1<d){d=b1;gId=2;}
  vec2 par=mod(gCell,2.);
  float a2=asinh(min(abs(X-par.x*uS),abs(Y-par.y*uS))*w), dc=lz-LN2;
  float b2=sqrt(dc*dc+a2*a2)-uTube; if(b2<d){d=b2;gId=2;}
  vec3 lc=vec3(uS*.5,uS*.5,1.41421356);
  vec3 dq=vec3(q,Z)-lc;
  float dl=acosh(1.+dot(dq,dq)/(2.*Z*lc.z))-0.05; if(dl<d){d=dl;gId=3;}
  return d;
}
#include "normal.glsl"

void glassFloor(vec4 p, vec4 v, inout vec3 thr, inout vec3 acc, float t){
  if(uGlass<.5) return;
  vec2 par=mod(gCell,2.); bool odd=mod(par.x+par.y,2.)>.5;
  float fr=fresnel(abs(md(v,tp(vec4(0,0,1,1),p)))); float fg=exp(-t/uFog);
  vec3 tint=odd? vec3(0.78,0.86,0.97): vec3(0.98,0.84,0.62);
  acc+=thr*fg*fr*tint*0.2; thr*=mix(vec3(1.),tint,0.28)*(1.-0.35*fr);
}
void glassWall(vec4 p, vec4 v, vec4 nn, inout vec3 thr, inout vec3 acc, float t){
  if(uGlass<.5) return;
  float fr=fresnel(abs(md(v,tp(nn,p)))); float fg=exp(-t/uFog);
  acc+=thr*fg*fr*vec3(0.30,0.48,0.85)*0.16; thr*=vec3(0.95,0.96,0.985)*(1.-0.35*fr);
}
void foldH(inout vec4 p, inout vec4 v, inout vec3 thr, inout vec3 acc, float t){
  for(int it=0;it<12;it++){
    float w=p.w-p.z, Z=1./w;
    if(Z>=2.){ dil(p,-LN2); dil(v,-LN2); vec2 par=mod(gCell,2.); gCell=floor(gCell*.5);
      parab(p,par.x*uS*.5,par.y*uS*.5); parab(v,par.x*uS*.5,par.y*uS*.5); glassFloor(p,v,thr,acc,t); continue; }
    if(Z<1.){ glassFloor(p,v,thr,acc,t); dil(p,LN2); dil(v,LN2); gCell*=2.; continue; }
    float X=p.x*Z, Y=p.y*Z;
    if(X>=uS){ parab(p,-uS,0.); parab(v,-uS,0.); gCell.x+=1.; glassWall(p,v,vec4(1,0,0,0),thr,acc,t); continue; }
    if(X<0.){ glassWall(p,v,vec4(1,0,0,0),thr,acc,t); parab(p,uS,0.); parab(v,uS,0.); gCell.x-=1.; continue; }
    if(Y>=uS){ parab(p,0.,-uS); parab(v,0.,-uS); gCell.y+=1.; glassWall(p,v,vec4(0,1,0,0),thr,acc,t); continue; }
    if(Y<0.){ glassWall(p,v,vec4(0,1,0,0),thr,acc,t); parab(p,0.,uS); parab(v,0.,uS); gCell.y-=1.; continue; }
    break;
  }
  gCell=mod(gCell,1024.);
}
void main(){
  vec2 uv=(2.*gl_FragCoord.xy-uRes)/uRes.y;
  vec3 dr=normalize(vec3(uv*uTanFov,1.));
  vec4 p=uCam[3], v=uCam[0]*dr.x+uCam[1]*dr.y+uCam[2]*dr.z;
  gCell=uCell;
  vec3 thr=vec3(1.), acc=vec3(0.); float t=0., d=0.; bool hit=false;
  for(int i=0;i<180;i++){
    foldH(p,v,thr,acc,t);
    d=map(p);
    if(d<3e-4*(1.+t)){hit=true;break;}
    float lz=log(1./(p.w-p.z));
    float s=min(d,min(lz,LN2-lz)+0.015);
    adv(p,v,s); t+=s;
    if(t>14.||thr.g<0.02) break;
  }
  vec3 col=FOG;
  if(hit){
    int id=gId; vec4 n=calcN(p,d);
    vec3 lamp=vec3(1.0,0.76,0.46);
    vec3 sc;
    if(id==3) sc=lamp*2.4;
    else{
      vec4 Lp=fromHS(vec3(uS*.5,uS*.5,1.41421356));
      vec4 L=tp(Lp,p); float sr=sinh(acosh(max(-md(p,Lp),1.)));
      float att=1.3/(0.2+sr*sr);
      float dif=max(md(n,L),0.), head=max(-md(n,v),0.);
      vec4 hv=tp(L-v,p); float sp=pow(max(md(n,hv),0.),40.);
      vec3 base=id==1? vec3(0.82,0.58,0.28): vec3(0.62,0.44,0.24);
      sc=base*(0.06+dif*att*lamp+0.24*head)+sp*att*lamp*0.45;
    }
    col=mix(FOG,sc,exp(-t/uFog));
  }
  fragColor=vec4(finish(acc+thr*col),1.);
}
