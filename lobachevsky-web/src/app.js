// Общее изменяемое состояние: текущий мир, камеры миров, масштаб рендера, ввод.
export const app={world:null,state:{},scale:matchMedia('(pointer:coarse)').matches?0.45:0.7,frameAvg:1/60};
export const inp={keys:new Set(),look:[0,0],joy:[0,0],vbtn:0,moved:false};
if(typeof window!=='undefined')window.__lob={app,inp};
