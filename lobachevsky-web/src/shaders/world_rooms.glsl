#include "base.glsl"

uniform vec4 uM[4]; uniform vec4 uE1; uniform vec4 uE2; uniform vec4 uV;
uniform float uHasV; uniform float uBall; uniform float uParity; uniform float uCamPar;
const vec3 FOG=vec3(0.028,0.032,0.066);
float map(vec4 p){
  float a=md(p,uE1), b=md(p,uE2);
  float d=asinh(sqrt(a*a+b*b))-uTube; gId=1;
  if(uHasV>.5){ float dv=acosh(max(-md(p,uV),1.))-uBall; if(dv<d){d=dv;gId=2;} }
  float dl=acosh(max(p.w,1.))-0.055; if(dl<d){d=dl;gId=3;}
  return d;
}
#include "normal.glsl"

void main(){
  vec2 uv=(2.*gl_FragCoord.xy-uRes)/uRes.y;
  vec3 dr=normalize(vec3(uv*uTanFov,1.));
  vec4 p=uCam[3], v=uCam[0]*dr.x+uCam[1]*dr.y+uCam[2]*dr.z;
  vec3 thr=vec3(1.), acc=vec3(0.); float t=0., d=0.; int cells=0; bool hit=false;
  for(int i=0;i<160;i++){
    for(int it=0;it<16;it++){
      bool any=false;
      for(int k=0;k<4;k++){
        float s=md(p,uM[k]);
        if(s>0.){
          p-=2.*s*uM[k]; v-=2.*md(v,uM[k])*uM[k]; any=true;
          if(k==0){ cells++;
            if(uGlass>.5){ float fr=fresnel(abs(md(v,uM[0]))); float fg=exp(-t/uFog);
              acc+=thr*fg*fr*vec3(0.30,0.48,0.85)*0.22; thr*=vec3(0.93,0.95,0.98)*(1.-0.35*fr); }
          }
        }
      }
      if(!any) break;
    }
    d=map(p);
    if(d<3e-4*(1.+t)){hit=true;break;}
    adv(p,v,d); t+=d;
    if(t>15.||thr.g<0.02) break;
  }
  vec3 col=FOG;
  if(hit){
    int id=gId; vec4 n=calcN(p,d);
    bool odd=uParity>.5 && ((cells+int(uCamPar))%2==1);
    vec3 lamp=odd? vec3(0.45,0.78,1.0): vec3(1.0,0.72,0.40);
    vec3 sc;
    if(id==3) sc=lamp*2.4;
    else{
      vec4 L=tp(vec4(0,0,0,1),p); float sr=sinh(acosh(max(p.w,1.)));
      float att=1.5/(0.22+sr*sr);
      float dif=max(md(n,L),0.), head=max(-md(n,v),0.);
      vec4 hv=tp(L-v,p); float sp=pow(max(md(n,hv),0.),48.);
      vec3 base=id==1? vec3(0.80,0.55,0.26): vec3(0.88,0.85,0.80);
      sc=base*(0.05+dif*att*lamp+0.22*head)+sp*att*lamp*0.5;
    }
    col=mix(FOG,sc,exp(-t/uFog));
  }
  fragColor=vec4(finish(acc+thr*col),1.);
}
