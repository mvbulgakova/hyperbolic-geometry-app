#include "base.glsl"

uniform vec4 uQ[40]; uniform vec4 uT1[40]; uniform vec4 uT2[40]; uniform vec4 uI[40];
uniform float uNf; uniform vec4 uSun;
uniform vec4 uGQ[4]; uniform vec4 uGF[4]; uniform float uGn; uniform float uConeC; uniform float uConeR; uniform float uDay;
const vec3 HAZE=vec3(0.66,0.76,0.90);
float h21(vec2 p){ p=fract(p*vec2(123.34,456.21)); p+=dot(p,p+45.32); return fract(p.x*p.y); }
float vn(vec2 p){ vec2 i=floor(p),f=fract(p); f=f*f*(3.-2.*f);
  return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y); }
vec4 lift(vec4 q, float h){ return cosh(h)*q+sinh(h)*vec4(0,0,1,0); }
float map(vec4 p){
  float hgt=asinh(p.z); float d=hgt; gId=0; int n=int(uNf+.5);
  for(int i=0;i<40;i++){
    if(i>=n) break;
    vec4 q=uQ[i]; float R=uI[i].w;
    float dq=acosh(max(-md(p,q),1.));
    if(dq-R>d) continue;
    float ty=uI[i].x, sz=uI[i].y;
    float a=md(p,uT1[i]), b=md(p,uT2[i]);
    float cyl=asinh(sqrt(a*a+b*b));
    if(ty<1.5){
      float tr=max(cyl-0.03*sz, hgt-0.20*sz); if(tr<d){d=tr;gId=1;}
      float cr=acosh(max(-md(p,lift(q,0.30*sz)),1.))-0.17*sz; if(cr<d){d=cr;gId=2;}
    }else if(ty<2.5){
      float w=max(cyl-0.15*sz, hgt-0.13*sz); if(w<d){d=w;gId=3;}
      float r=max(acosh(max(-md(p,lift(q,0.09*sz)),1.))-0.20*sz, 0.13*sz-hgt); if(r<d){d=r;gId=4;}
    }else if(ty<3.5){
      float pl=max(cyl-0.008, hgt-0.26); if(pl<d){d=pl;gId=5;}
      vec4 q2=cosh(0.04)*q+sinh(0.04)*uT1[i];
      float pn=acosh(max(-md(p,lift(q2,0.23)),1.))-0.035; if(pn<d){d=pn;gId=6;}
    }else if(ty<4.5){
      float rk=acosh(max(-md(p,lift(q,0.02*sz)),1.))-0.07*sz; if(rk<d){d=rk;gId=7;}
    }else if(ty<5.5){
      float bd=max(cyl-0.022, hgt-0.075); if(bd<d){d=bd;gId=8;}
      float hd=acosh(max(-md(p,lift(q,0.092)),1.))-0.018; if(hd<d){d=hd;gId=9;}
      vec4 q3=cosh(0.035)*q+sinh(0.035)*uT1[i];
      float ld=acosh(max(-md(p,lift(q3,0.055)),1.))-0.011; if(ld<d){d=ld;gId=10;}
    }else{
      float ch=acosh(max(-md(p,lift(q,0.025)),1.))-0.03; if(ch<d){d=ch;gId=11;}
    }
  }
  return d*0.85;
}
#include "normal.glsl"

vec3 sky(vec4 p, vec4 v){
  vec4 up=tp(vec4(0,0,1,0),p); float sv=md(v,up);
  vec3 c=mix(HAZE, vec3(0.20,0.42,0.82), smoothstep(0.0,0.75,sv));
  vec4 s=tp(uSun,p); float cs=md(v,s);
  c+=vec3(1.0,0.82,0.55)*pow(max(cs,0.),48.)*0.45;
  c=mix(c,vec3(1.3,1.15,0.9),smoothstep(0.9990,0.9996,cs));
  return c;
}
float shadow(vec4 p0){
  int sid=gId; vec4 p=p0; vec4 v=tp(uSun,p0); float res=1., tt=0.;
  adv(p,v,0.03); tt=0.03;
  for(int i=0;i<36;i++){
    float dd=map(p); res=min(res,7.*dd/tt);
    if(res<0.02||tt>2.5) break;
    float st=max(dd,0.012); adv(p,v,st); tt+=st;
  }
  gId=sid; return clamp(res,0.,1.);
}
void main(){
  vec2 uv=(2.*gl_FragCoord.xy-uRes)/uRes.y;
  vec3 dr=normalize(vec3(uv*uTanFov,1.));
  vec4 p=uCam[3], v=uCam[0]*dr.x+uCam[1]*dr.y+uCam[2]*dr.z;
  vec4 p0=p, v0=v;
  float t=0., d=0.; bool hit=false;
  for(int i=0;i<150;i++){
    d=map(p);
    if(d<3e-4*(1.+t)){hit=true;break;}
    adv(p,v,d); t+=d;
    if(t>13.||p.w>2.0e4) break;
  }
  float dayK=mix(0.22,1.,uDay);
  vec3 col=sky(p0,v0)*dayK;
  if(hit){
    int id=gId; vec4 n=calcN(p,d);
    vec4 up=tp(vec4(0,0,1,0),p); vec4 s=tp(uSun,p);
    vec3 base; float spec=0.;
    if(id==0){
      float ex=abs(asinh(p.x)), ey=abs(asinh(p.y)), er=min(ex,ey);
      float far=exp(-t*0.8);
      float nz=(vn(p.xy*9.)*0.6+vn(p.xy*31.)*0.4);
      vec3 grass=mix(vec3(0.17,0.34,0.09),vec3(0.30,0.46,0.14),mix(0.5,nz,far));
      float road=1.-smoothstep(0.075,0.095,er);
      vec3 dirt=mix(vec3(0.47,0.38,0.27),vec3(0.58,0.50,0.37),mix(0.5,vn(p.xy*24.),far));
      float edge=smoothstep(0.065,0.08,er)*(1.-smoothstep(0.08,0.10,er));
      base=mix(grass,dirt,road); base*=1.-0.25*edge;
    }
    else if(id==1) base=vec3(0.33,0.21,0.12);
    else if(id==2) base=mix(vec3(0.10,0.28,0.08),vec3(0.22,0.42,0.12),vn(p.xy*37.+p.z*23.));
    else if(id==3) base=vec3(0.88,0.82,0.70);
    else if(id==4) base=vec3(0.62,0.21,0.14);
    else if(id==5) base=vec3(0.92,0.92,0.90);
    else if(id==6) base=vec3(0.90,0.18,0.12);
    else if(id==7) base=vec3(0.50,0.50,0.48)*(0.8+0.4*vn(p.xy*17.));
    else if(id==8) base=vec3(0.13,0.13,0.17);
    else if(id==9) base=vec3(0.80,0.64,0.50);
    else if(id==10) base=vec3(1.0,0.8,0.4);
    else base=vec3(0.95,0.72,0.22);
    float sh=shadow(p);
    float dif=max(md(n,s),0.)*sh;
    float amb=0.30+0.70*max(md(n,up),0.)*0.5;
    vec3 sunc=vec3(1.0,0.90,0.72)*2.0*uDay*uDay, skyc=vec3(0.42,0.56,0.85)*dayK;
    vec3 sc=base*(sunc*dif+skyc*amb*0.55);
    if(id==6) sc+=base*0.6;
    if(id==10) sc=vec3(1.8,1.3,0.6);
    if(id==11) sc+=base*0.45;
    if(id==0){ float lg=0.;
      for(int g=0;g<4;g++){ if(float(g)>=uGn) break;
        float dg=acosh(max(-md(p,uGQ[g]),1.));
        if(dg<uConeR){ vec4 u=tp(p,uGQ[g]); float cc=md(u,uGF[g]);
          lg+=smoothstep(uConeC-0.015,uConeC+0.015,cc)*(1.-dg/uConeR); } }
      sc+=base*vec3(1.0,0.78,0.42)*lg*mix(4.5,1.6,uDay); }
    col=mix(HAZE*dayK,sc,exp(-t/uFog));
  }
  fragColor=vec4(finish(col),1.);
}
