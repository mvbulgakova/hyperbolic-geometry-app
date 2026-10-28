#include "base.glsl"

uniform vec4 uM[3]; uniform vec4 uE1; uniform vec4 uE2; uniform float uCamPar; uniform float uLampH;
const vec3 FOG=vec3(0.20,0.21,0.30);
float map(vec4 p){
  float d=asinh(p.z); gId=0;
  float a=md(p,uE1), b=md(p,uE2);
  float dp=asinh(sqrt(a*a+b*b))-uTube; if(dp<d){d=dp;gId=1;}
  vec4 C=vec4(0.,0.,sinh(uLampH),cosh(uLampH));
  float dl=acosh(max(-md(p,C),1.))-0.07; if(dl<d){d=dl;gId=3;}
  return d;
}
#include "normal.glsl"

void main(){
  vec2 uv=(2.*gl_FragCoord.xy-uRes)/uRes.y;
  vec3 dr=normalize(vec3(uv*uTanFov,1.));
  vec4 p=uCam[3], v=uCam[0]*dr.x+uCam[1]*dr.y+uCam[2]*dr.z;
  float t=0., d=0.; int par=0; bool hit=false;
  for(int i=0;i<170;i++){
    for(int it=0;it<16;it++){
      bool any=false;
      for(int k=0;k<3;k++){ float s=md(p,uM[k]);
        if(s>0.){ p-=2.*s*uM[k]; v-=2.*md(v,uM[k])*uM[k]; any=true; if(k==0) par++; } }
      if(!any) break;
    }
    d=map(p);
    if(d<3e-4*(1.+t)){hit=true;break;}
    adv(p,v,d); t+=d;
    if(t>18.) break;
  }
  vec4 up=tp(vec4(0,0,1,0),p);
  float sv=md(v,up);
  vec3 sky=mix(vec3(0.20,0.21,0.30),vec3(0.035,0.045,0.10),smoothstep(-0.1,0.9,sv));
  vec3 col=sky;
  if(hit){
    int id=gId; vec4 n=calcN(p,d);
    vec3 lamp=vec3(1.0,0.78,0.48);
    vec4 C=vec4(0.,0.,sinh(uLampH),cosh(uLampH));
    vec4 L=tp(C,p); float sr=sinh(acosh(max(-md(p,C),1.)));
    float att=0.9/(0.15+sr*sr);
    float dif=max(md(n,L),0.), skyl=0.35+0.65*max(md(n,up),0.), head=max(-md(n,v),0.);
    vec3 sc;
    if(id==3) sc=lamp*2.4;
    else if(id==1){ vec3 base=vec3(0.80,0.56,0.27); vec4 hv=tp(L-v,p);
      sc=base*(0.18*skyl+dif*att*lamp+0.2*head)+pow(max(md(n,hv),0.),40.)*att*lamp*0.5; }
    else{
      bool odd=((par+int(uCamPar))%2)==1;
      vec3 base=odd? vec3(0.22,0.32,0.44): vec3(0.84,0.77,0.63);
      float e=asinh(abs(md(p,uM[0])));
      float lw=0.010+0.0025*t;
      base=mix(base,vec3(0.07,0.06,0.05),1.-smoothstep(lw,lw*1.8,e));
      sc=base*(0.22*skyl+dif*att*lamp*0.9);
    }
    col=mix(FOG,sc,exp(-t/uFog));
  }
  fragColor=vec4(finish(col),1.);
}
