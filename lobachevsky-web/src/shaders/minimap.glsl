#version 300 es
precision highp float;
uniform vec2 uRes; uniform mat4 uT; uniform vec4 uM[3]; uniform vec4 uV; uniform float uCamPar; uniform float uFovH;
out vec4 fragColor;
float md(vec4 a, vec4 b){ return dot(a.xyz,b.xyz)-a.w*b.w; }
void main(){
  vec2 uv=(2.*gl_FragCoord.xy-uRes)/uRes.y*1.04;
  float rr=length(uv);
  if(rr>=0.998){ float ring=1.-smoothstep(0.004,0.014,abs(rr-1.003)); fragColor=vec4(vec3(0.84,0.64,0.35)*ring,ring); return; }
  vec4 q=uT*(vec4(uv,0.,1.)*inversesqrt(1.-rr*rr));
  int par=0;
  for(int it=0;it<64;it++){ bool any=false;
    for(int k=0;k<3;k++){ float s=md(q,uM[k]); if(s>0.){ q-=2.*s*uM[k]; any=true; if(k==0) par++; } }
    if(!any) break; }
  bool odd=((par+int(uCamPar))%2)==1;
  vec3 c=odd? vec3(0.22,0.32,0.44): vec3(0.84,0.77,0.63);
  float e=asinh(abs(md(q,uM[0]))); float fw=fwidth(e);
  c=mix(c,vec3(0.07,0.06,0.05),1.-smoothstep(fw*0.5,fw*1.5,e));
  float dv=acosh(max(-md(q,uV),1.)); float fv=fwidth(dv);
  c=mix(c,vec3(0.86,0.60,0.30),1.-smoothstep(0.06,0.06+fv*1.5,dv));
  c=mix(c,vec3(0.06,0.07,0.12),smoothstep(0.82,1.0,rr)*0.55);
  float ang=atan(uv.x,uv.y);
  if(rr<0.34 && abs(ang)<uFovH*.5) c=mix(c,vec3(1.,0.96,0.82),0.32*(1.-rr/0.34));
  c=mix(c,vec3(0.95,0.36,0.22),1.-smoothstep(0.022,0.034,rr));
  fragColor=vec4(c,1.);
}
