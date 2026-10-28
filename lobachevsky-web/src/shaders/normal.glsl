vec4 calcN(vec4 p, float d0){
  int sid=gId;
  vec4 t0=tp(vec4(1,0,0,0),p);
  vec4 t1=vec4(0,1,0,0); t1+=md(t1,p)*p; t1-=md(t1,t0)*t0; t1*=inversesqrt(max(md(t1,t1),1e-20));
  vec4 t2=vec4(0,0,1,0); t2+=md(t2,p)*p; t2-=md(t2,t0)*t0; t2-=md(t2,t1)*t1; t2*=inversesqrt(max(md(t2,t2),1e-20));
  const float e=1e-3;
  vec4 n=(map(p+e*t0)-d0)*t0+(map(p+e*t1)-d0)*t1+(map(p+e*t2)-d0)*t2;
  gId=sid; return tp(n,p);
}
