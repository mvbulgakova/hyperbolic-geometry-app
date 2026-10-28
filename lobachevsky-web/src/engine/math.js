

// ───────────────────────── Линейная алгебра в R^{3,1} ─────────────────────────
// Векторы (x1,x2,x3,x0), ⟨x,y⟩ = x1y1 + x2y2 + x3y3 − x0y0; матрицы 4×4 хранятся по столбцам.
export const md=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2]-a[3]*b[3];
export const I4=()=>[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
export const O=[0,0,0,1];
export function mul(A,B){const C=new Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=A[k*4+r]*B[c*4+k];C[c*4+r]=s;}return C;}
export function apply(A,v){const o=[0,0,0,0];for(let r=0;r<4;r++)o[r]=A[r]*v[0]+A[4+r]*v[1]+A[8+r]*v[2]+A[12+r]*v[3];return o;}
export const col=(M,c)=>M.slice(c*4,c*4+4);
export function setCol(M,c,v){for(let r=0;r<4;r++)M[c*4+r]=v[r];}
export function refl(n){const Jn=[n[0],n[1],n[2],-n[3]],R=new Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)R[c*4+r]=(r===c?1:0)-2*n[r]*Jn[c];return R;}
export function boostLocal(M,k,s){const ch=Math.cosh(s),sh=Math.sinh(s),a=col(M,k),b=col(M,3);setCol(M,k,a.map((x,i)=>ch*x+sh*b[i]));setCol(M,3,b.map((x,i)=>sh*a[i]+ch*x));}
export function rotLocal(M,i,j,t){const c=Math.cos(t),s=Math.sin(t),a=col(M,i),b=col(M,j);setCol(M,i,a.map((x,q)=>c*x+s*b[q]));setCol(M,j,b.map((x,q)=>-s*a[q]+c*x));}
export function gs(M){
  let p=col(M,3);const k=1/Math.sqrt(Math.max(-md(p,p),1e-30));p=p.map(x=>x*k);setCol(M,3,p);
  const cs=[];
  for(let i=0;i<3;i++){let v=col(M,i);const d=md(v,p);v=v.map((x,q)=>x+d*p[q]);
    for(const u of cs){const e=md(v,u);v=v.map((x,q)=>x-e*u[q]);}
    const n=1/Math.sqrt(Math.max(md(v,v),1e-30));v=v.map(x=>x*n);cs.push(v);setCol(M,i,v);}
}
export function rowsToCols(R){const M=new Array(16);for(let r=0;r<4;r++)for(let c=0;c<4;c++)M[c*4+r]=R[r][c];return M;}
export function parabM(a,b){const s=a*a+b*b;return rowsToCols([[1,0,-a,a],[0,1,-b,b],[a,b,1-s/2,s/2],[a,b,-s/2,1+s/2]]);}
export function boostZ(h){const c=Math.cosh(h),s=Math.sinh(h);return rowsToCols([[1,0,0,0],[0,1,0,0],[0,0,c,s],[0,0,s,c]]);}
export function boostX(h){const c=Math.cosh(h),s=Math.sinh(h);return rowsToCols([[c,0,0,s],[0,1,0,0],[0,0,1,0],[s,0,0,c]]);}
export function rotZ(t){const c=Math.cos(t),s=Math.sin(t);return rowsToCols([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]]);}
export function rotYP(y,p){const cy=Math.cos(y),sy=Math.sin(y),cp=Math.cos(p),sp=Math.sin(p);
  return [sy,-cy,0,0, -cy*sp,-sy*sp,cp,0, cy*cp,sy*cp,sp,0, 0,0,0,1];}
export function det3(a,b,c,d,e,f,g,h,i){return a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);}
export function planeThrough(A,B,C){ // нормаль n: ⟨n,A⟩ = ⟨n,B⟩ = ⟨n,C⟩ = 0
  const m=[A,B,C],w=[0,0,0,0];
  for(let i=0;i<4;i++){const cols=[0,1,2,3].filter(j=>j!==i);
    w[i]=(i%2?-1:1)*det3(...cols.map(j=>m[0][j]),...cols.map(j=>m[1][j]),...cols.map(j=>m[2][j]));}
  const n=[w[0],w[1],w[2],-w[3]],k=1/Math.sqrt(md(n,n));return n.map(x=>x*k);
}
export function solve(G,b){const n=b.length,A=G.map((r,i)=>[...r,b[i]]);
  for(let i=0;i<n;i++){let p=i;for(let r=i+1;r<n;r++)if(Math.abs(A[r][i])>Math.abs(A[p][i]))p=r;[A[i],A[p]]=[A[p],A[i]];
    for(let r=0;r<n;r++)if(r!==i){const f=A[r][i]/A[i][i];for(let c=i;c<=n;c++)A[r][c]-=f*A[i][c];}}
  return A.map((r,i)=>r[n]/r[i]);}
export function foot(ns){ // ближайшая к центру точка пересечения плоскостей
  const a=solve(ns.map(x=>ns.map(y=>md(x,y))),ns.map(n=>n[3]));
  const x=O.slice();ns.forEach((n,j)=>{for(let q=0;q<4;q++)x[q]+=a[j]*n[q];});
  const nn=md(x,x);if(nn<-1e-9){const k=1/Math.sqrt(-nn);return x.map(v=>v*k);}return x[3]<0?x.map(v=>-v):x;
}
export function orth(a,b){const d=md(a,b),v=a.map((x,i)=>x-d*b[i]),k=1/Math.sqrt(md(v,v));return v.map(x=>x*k);}
export function orient(n,P){return md(n,P)>0?n.map(x=>-x):n;}

export function tanDir(A,B){const d=md(B,A),v=B.map((x,i)=>x+d*A[i]),k=1/Math.sqrt(Math.max(md(v,v),1e-30));return v.map(x=>x*k);}
export const hdist=(P,Q)=>Math.acosh(Math.max(1,-md(P,Q)));
