#version 300 es
precision highp float;
uniform vec2 uRes; uniform mat4 uCam; uniform float uTanFov; uniform float uFog; uniform float uGlass; uniform float uTube;
out vec4 fragColor;
int gId;
float md(vec4 a, vec4 b){ return dot(a.xyz,b.xyz)-a.w*b.w; }
vec4 hn(vec4 p){ return p*inversesqrt(max(-md(p,p),1e-20)); }
vec4 tp(vec4 v, vec4 p){ v+=md(v,p)*p; return v*inversesqrt(max(md(v,v),1e-20)); }
void adv(inout vec4 p, inout vec4 v, float s){ float c=cosh(s), h=sinh(s); vec4 np=c*p+h*v; v=h*p+c*v; p=hn(np); v=tp(v,p); }
float fresnel(float c){ return 0.04+0.96*pow(1.-clamp(c,0.,1.),5.); }
vec3 finish(vec3 c){ c=1.-exp(-c*1.15); c=pow(c,vec3(1./2.2));
  vec2 q=gl_FragCoord.xy/uRes-.5; c*=1.-0.35*dot(q,q); return c; }
