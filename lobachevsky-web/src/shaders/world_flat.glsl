#version 300 es
precision highp float;
uniform vec2 uRes; uniform mat4 uCam; uniform float uTanFov; uniform float uFog; uniform float uGlass; uniform float uTube; uniform float uBall;
out vec4 fragColor;
const vec3 FOG=vec3(0.028,0.032,0.066);
const float S=1.0;
int gId;
float map(vec3 p){ vec3 f=fract(p/S), m=min(f,1.-f);
  float d=min(min(length(m.xy),length(m.yz)),length(m.xz))*S-uTube; gId=1;
  float db=length(m)*S-uBall; if(db<d){d=db;gId=2;}
  float dl=length(f-.5)*S-0.055; if(dl<d){d=dl;gId=3;}
  return d; }
float fresnel(float c){ return 0.04+0.96*pow(1.-clamp(c,0.,1.),5.); }
void main(){
  vec2 uv=(2.*gl_FragCoord.xy-uRes)/uRes.y;
  vec3 dr=normalize(vec3(uv*uTanFov,1.));
  vec3 p=uCam[3].xyz, v=normalize(uCam[0].xyz*dr.x+uCam[1].xyz*dr.y+uCam[2].xyz*dr.z);
  vec3 thr=vec3(1.), acc=vec3(0.); float t=0., d=0.; bool hit=false;
  vec3 cell=floor(p/S);
  for(int i=0;i<160;i++){
    d=map(p);
    if(d<3e-4*(1.+t)){hit=true;break;}
    p+=v*d; t+=d;
    vec3 nc=floor(p/S);
    if(nc!=cell){
      vec3 ch=abs(nc-cell);
      for(int a=0;a<3;a++) if(ch[a]>0.5 && uGlass>.5){ float fr=fresnel(abs(v[a])); float fg=exp(-t/uFog);
        acc+=thr*fg*fr*vec3(0.30,0.48,0.85)*0.22; thr*=vec3(0.93,0.95,0.98)*(1.-0.35*fr); }
      cell=nc;
    }
    if(t>15.||thr.g<0.02) break;
  }
  vec3 col=FOG;
  if(hit){
    int id=gId; const float e=1e-3;
    vec3 n=normalize(vec3(map(p+vec3(e,0,0))-d,map(p+vec3(0,e,0))-d,map(p+vec3(0,0,e))-d)); gId=id;
    vec3 c=(floor(p/S)+.5)*S;
    bool odd=mod(floor(p.x/S)+floor(p.y/S)+floor(p.z/S),2.)>.5;
    vec3 lamp=odd? vec3(0.45,0.78,1.0): vec3(1.0,0.72,0.40);
    vec3 sc;
    if(id==3) sc=lamp*2.4;
    else{ vec3 L=normalize(c-p); float r=length(c-p); float att=1.5/(0.22+r*r);
      float dif=max(dot(n,L),0.), head=max(-dot(n,v),0.); vec3 hv=normalize(L-v); float sp=pow(max(dot(n,hv),0.),48.);
      vec3 base=id==1? vec3(0.80,0.55,0.26): vec3(0.88,0.85,0.80);
      sc=base*(0.05+dif*att*lamp+0.22*head)+sp*att*lamp*0.5; }
    col=mix(FOG,sc,exp(-t/uFog));
  }
  col=acc+thr*col; col=1.-exp(-col*1.15); col=pow(col,vec3(1./2.2));
  vec2 q=gl_FragCoord.xy/uRes-.5; col*=1.-0.35*dot(q,q);
  fragColor=vec4(col,1.);
}
