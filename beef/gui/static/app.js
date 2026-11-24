"use strict";
/* ============================================================= *
 *  BEEF GUI — front end (model editing, viewer, results); analyses run in the local Python server
 * ============================================================= */
/* ===================== State & utilities ===================== */
const $ = (s,el=document)=>el.querySelector(s);
const $$ = (s,el=document)=>[...el.querySelectorAll(s)];
let UID = 1; const uid = ()=> UID++;
const clone = o=>JSON.parse(JSON.stringify(o));
const num = (v,d=0)=>{ const n=parseFloat(v); return isFinite(n)?n:d; };
function toast(msg,type=''){ const t=$('#toast'); t.textContent=msg; t.className='toast show '+type; clearTimeout(t._t); t._t=setTimeout(()=>t.className='toast',2600); }

/* Save text as a download in the browser */
function saveFile(filename, text, mime='text/plain'){
  const blob=new Blob([text],{type:mime});
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
  toast('Downloaded '+filename,'ok');
}
function showTextModal(title,text){
  const wrap=document.createElement('div'); wrap.className='modal show';
  wrap.innerHTML=`<div class="box"><button class="closex">×</button><h2>${esc(title)}</h2><p>Copy the contents below.</p><textarea style="width:100%;height:50vh">${esc(text)}</textarea></div>`;
  document.body.appendChild(wrap);
  wrap.querySelector('.closex').onclick=()=>wrap.remove();
  wrap.addEventListener('mousedown',e=>{if(e.target===wrap)wrap.remove();});
  wrap.querySelector('textarea').select();
}

// The working model
let M = null;
function blankModel(domain='3d'){
  return {
    domain,
    nodes:[],      // {id,label,x,y,z}
    elements:[],   // {id,label,n1,n2,section,etype,part}
    sections:[],   // {id,name,E,rho,A,I_y,I_z,J,poisson,m,G,kappa,shear,massform}
    constraints:[],// {id,name,masters:'',slaves:'',dofs:'all'}
    features:[],   // {id,ftype,nodes:'',dofs:'',value}
    forces:[],     // {id,name,nodes:'',dof:1,amp,mode,texpr,case}
    element_loads:[], // {id,name,target:'all'|'list'|'section',elements,section,option,value,fun,g,direction,include_moments,mode,case}
    analysis:{ kind:'lin_static', tmax:1, dt:1, t0:0, itmax:10, tolU:'', tolR:'',
               rMass:0, rStiff:0, beta:0.25, gamma:0.5, alpha:0, nModes:'',
               include_kg:false, scale:1 }
  };
}
const ndofs = ()=> M.domain==='2d'?3:6;
const nodeTypeStr = ()=> M.domain==='2d'?'beam2d':'beam3d';
const DOFNAMES = { '2d':['ux','uy','rz'], '3d':['ux','uy','uz','rx','ry','rz'] };

/* parse "1,2,5-8" -> [1,2,5,6,7,8] */
function parseInts(s){
  if(s===null||s===undefined) return [];
  const out=[];
  String(s).split(',').forEach(part=>{
    part=part.trim(); if(!part) return;
    const m=part.match(/^(-?\d+)\s*-\s*(-?\d+)$/);
    if(m){ let a=+m[1],b=+m[2],st=a<=b?1:-1; for(let i=a;i!=b+st;i+=st) out.push(i); }
    else if(/^-?\d+$/.test(part)) out.push(+part);
  });
  return out;
}
/* matrix text ("a, b; c, d" or one row per line) -> [[a,b],[c,d]] */
function parseMatrix(s){
  return String(s||'').split(/[\n;]+/).map(r=>r.trim()).filter(Boolean)
    .map(r=>r.split(/[\s,]+/).filter(Boolean).map(Number));
}
function matrixText(mat){ return (mat||[]).map(r=>r.map(v=>String(v)).join(', ')).join('\n'); }
function parseDofs(s){
  const t=String(s).trim().toLowerCase();
  if(t==='all'||t==='trans'||t==='rot') return t;
  return parseInts(s);
}

/* ===================== Tabs ===================== */
$$('#ltabs .tab').forEach(tab=>tab.addEventListener('click',()=>{
  $$('#ltabs .tab').forEach(t=>t.classList.remove('active'));
  $$('.panes .pane').forEach(p=>p.classList.remove('active'));
  tab.classList.add('active');
  $('#'+tab.dataset.pane).classList.add('active');
}));

/* ===================== Generic editable list helper ===================== */
function bindNum(obj,key){ return e=>{ obj[key]= e.target.value===''? '' : e.target.value; scheduleDraw(); }; }

/* ===================== NODES ===================== */
/* Large models: only the first TABLE_LIMIT rows are rendered until "show all" is clicked */
const TABLE_LIMIT=200;
let SHOWALL={nodes:false, els:false};
function moreRow(tb,key,shown,total,ncol){
  if(total<=shown) return;
  const tr=document.createElement('tr');
  tr.innerHTML=`<td colspan="${ncol}" style="color:var(--muted);padding:6px">Showing ${shown} of ${total}. <button class="btn sm ghost">Show all</button></td>`;
  $('button',tr).onclick=()=>{ SHOWALL[key]=true; key==='nodes'?renderNodes():renderEls(); };
  tb.appendChild(tr);
}
function renderNodes(){
  const tb=$('#tNodes tbody'); tb.innerHTML='';
  const is2 = M.domain==='2d';
  $$('.zcol').forEach(e=>e.style.display=is2?'none':'');
  const list = SHOWALL.nodes? M.nodes : M.nodes.slice(0,TABLE_LIMIT);
  list.forEach(n=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td><input type="number" value="${n.label}" style="width:52px"></td>
      <td><input type="number" step="any" value="${n.x}" style="width:62px"></td>
      <td><input type="number" step="any" value="${n.y}" style="width:62px"></td>
      <td class="zcol" ${is2?'style="display:none"':''}><input type="number" step="any" value="${n.z}" style="width:62px"></td>
      <td><button class="delx">×</button></td>`;
    const [li,xi,yi,zi]=$$('input',tr);
    li.onchange=e=>{ n.label=parseInt(e.target.value)||n.label; refreshNodeRefs(); redrawAll(); };
    xi.oninput=e=>{ n.x=num(e.target.value); scheduleDraw(); };
    yi.oninput=e=>{ n.y=num(e.target.value); scheduleDraw(); };
    zi.oninput=e=>{ n.z=num(e.target.value); scheduleDraw(); };
    $('.delx',tr).onclick=()=>{ M.nodes=M.nodes.filter(x=>x!==n); redrawAll(); };
    tb.appendChild(tr);
  });
  moreRow(tb,'nodes',list.length,M.nodes.length,5);
  $('#nNodes').textContent=M.nodes.length;
}
function addNode(x=0,y=0,z=0,label){
  if(label===undefined) label=(M.nodes.reduce((m,n)=>Math.max(m,n.label),0)||0)+1;
  M.nodes.push({id:uid(),label,x,y,z});
}
$('#addNode').onclick=()=>{ addNode(); redrawAll(); };
$('#clrNodes').onclick=()=>{ if(confirm('Remove all nodes?')){ M.nodes=[]; redrawAll(); } };

/* ===================== ELEMENTS ===================== */
function secOptions(sel){ return M.sections.map(s=>`<option ${s.name===sel?'selected':''}>${esc(s.name)}</option>`).join(''); }
function partOptions(sel){
  const parts=[...new Set(M.elements.map(e=>e.part).concat(['1']))].sort();
  return parts.map(p=>`<option ${String(p)===String(sel)?'selected':''}>${esc(p)}</option>`).join('');
}
function renderEls(){
  const tb=$('#tEls tbody'); tb.innerHTML='';
  const src = ($('#selTable').checked && SEL.size)? M.elements.filter(e=>SEL.has(Number(e.label))) : M.elements;
  const list = SHOWALL.els? src : src.slice(0,TABLE_LIMIT);
  list.forEach(el=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td><input type="number" value="${el.label}" style="width:42px"></td>
      <td><input type="number" value="${el.n1}" style="width:42px"></td>
      <td><input type="number" value="${el.n2}" style="width:42px"></td>
      <td><select class="selSec" style="width:120px">${secOptions(el.section)}</select></td>
      <td><select class="selType" style="width:58px"><option ${el.etype==='beam'?'selected':''}>beam</option><option ${el.etype==='bar'?'selected':''}>bar</option></select></td>
      <td><input type="text" value="${esc(el.part)}" style="width:34px"></td>
      <td><button class="delx">×</button></td>`;
    const ins=$$('input',tr);
    ins[0].onchange=e=>{ el.label=parseInt(e.target.value)||el.label; redrawAll(); };
    ins[1].onchange=e=>{ el.n1=parseInt(e.target.value)||el.n1; scheduleDraw(); };
    ins[2].onchange=e=>{ el.n2=parseInt(e.target.value)||el.n2; scheduleDraw(); };
    ins[3].onchange=e=>{ el.part=e.target.value||'1'; scheduleDraw(); };
    $('.selSec',tr).onchange=e=>{ el.section=e.target.value; };
    $('.selType',tr).onchange=e=>{ el.etype=e.target.value; };
    $('.delx',tr).onclick=()=>{ M.elements=M.elements.filter(x=>x!==el); redrawAll(); };
    tb.appendChild(tr);
  });
  moreRow(tb,'els',list.length,src.length,7);
  $('#nEls').textContent=M.elements.length;
}
function addEl(n1,n2,label){
  if(label===undefined) label=(M.elements.reduce((m,e)=>Math.max(m,e.label),0)||0)+1;
  const sec=M.sections[0]?M.sections[0].name:'';
  if(n1===undefined){ const ls=M.nodes.map(n=>n.label); n1=ls[0]||1; n2=ls[1]||2; }
  M.elements.push({id:uid(),label,n1,n2,section:sec,etype:'beam',part:'1'});
}
$('#addEl').onclick=()=>{ if(!M.sections.length) addSection(); addEl(); redrawAll(); };
$('#clrEls').onclick=()=>{ if(confirm('Remove all elements?')){ M.elements=[]; redrawAll(); } };

/* chain existing nodes into elements in label order */
$('#chainEls').onclick=()=>{
  const seq=prompt('Node labels to chain (e.g. 1-5 or 1,2,3). Blank = all in current order:', '');
  if(seq===null) return;
  let labels = seq.trim()? parseInts(seq) : M.nodes.map(n=>n.label);
  if(labels.length<2){ toast('Need at least 2 nodes','err'); return; }
  if(!M.sections.length) addSection();
  for(let i=0;i<labels.length-1;i++) addEl(labels[i],labels[i+1]);
  redrawAll(); toast('Chained '+(labels.length-1)+' elements','ok');
};

/* generate a straight line of nodes (+optional elements) */
$('#genLine').onclick=()=>openLineGen();

/* ===================== SECTIONS ===================== */
function addSection(p={}){
  const name=p.name|| 'sec'+(M.sections.length+1);
  M.sections.push(Object.assign({id:uid(),name,E:210e9,rho:7850,A:0.01,I_y:1e-4,I_z:1e-4,J:1e-5,
    poisson:0.3,m:'',G:'',kappa:1.0,shear:false,massform:'euler',e2:''},p));
  return name;
}
/* exact, compact text for numeric fields: 210000000000 -> "2.1e11", 0.00008 -> "8e-5" (narrow fields cut long digit strings) */
function numText(v){
  if(v===''||v==null) return '';
  const x=Number(v); if(!isFinite(x)||typeof v==='string'&&isNaN(Number(v))) return String(v);
  const a=Math.abs(x);
  return (a!==0&&(a>=1e5||a<1e-3))? x.toExponential().replace('e+','e') : String(x);
}
function field(label,val,oninput,attrs='type="number" step="any"'){
  const w=document.createElement('div'); w.className='row';
  const txt= /type="number"/.test(attrs)? numText(val) : (val===''||val==null?'':String(val));
  w.innerHTML=`<label>${label}</label><input class="grow" ${attrs} value="${esc(txt)}">`;
  $('input',w).oninput=oninput; return w;
}
function renderSections(){
  const box=$('#secList'); box.innerHTML='';
  M.sections.forEach(s=>{
    const c=document.createElement('div'); c.className='ex'; c.style.cursor='default';
    const head=document.createElement('div'); head.className='row'; head.style.margin='0 0 6px';
    head.innerHTML=`<label>Name</label><input class="grow" type="text" value="${esc(s.name)}"><button class="delx">×</button>`;
    const ni=$('input',head);
    ni.onchange=e=>{ const old=s.name; s.name=e.target.value||old; M.elements.forEach(el=>{if(el.section===old)el.section=s.name;}); renderEls(); };
    $('.delx',head).onclick=()=>{ M.sections=M.sections.filter(x=>x!==s); renderSections(); renderEls(); };
    c.appendChild(head);
    const g=document.createElement('div'); g.style.display='grid'; g.style.gridTemplateColumns='1fr 1fr'; g.style.gap='0 8px';
    const is2=M.domain==='2d';
    g.appendChild(field('E (Pa)',s.E,e=>s.E=num(e.target.value)));
    g.appendChild(field('ρ (kg/m³)',s.rho,e=>s.rho=num(e.target.value)));
    g.appendChild(field('A (m²)',s.A,e=>s.A=num(e.target.value)));
    g.appendChild(field('I_y (m⁴)',s.I_y,e=>s.I_y=num(e.target.value)));
    if(!is2){
      g.appendChild(field('I_z (m⁴)',s.I_z,e=>s.I_z=e.target.value===''?'':num(e.target.value)));
      g.appendChild(field('J (m⁴)',s.J,e=>s.J=num(e.target.value)));
    }
    g.appendChild(field('Poisson ν',s.poisson,e=>s.poisson=num(e.target.value)));
    g.appendChild(field('m (kg/m, opt)',s.m,e=>s.m=e.target.value===''?'':num(e.target.value)));
    g.appendChild(field('G (Pa, opt)',s.G,e=>s.G=e.target.value===''?'':num(e.target.value)));
    g.appendChild(field('κ (shear)',s.kappa,e=>s.kappa=num(e.target.value)));
    c.appendChild(g);
    if(!is2){
      const r1=field('e2 (orient.)',s.e2||'',e=>s.e2=e.target.value.trim(),'type="text" placeholder="x,y,z — optional local y-axis"');
      c.appendChild(r1);
    }
    const r2=document.createElement('div'); r2.className='row';
    r2.innerHTML=`<label class="inline"><input type="checkbox" ${s.shear?'checked':''}> shear deform.</label>
      <label class="inline" style="margin-left:8px">mass
      <select><option ${s.massform==='euler'?'selected':''}>euler</option><option ${s.massform==='timoshenko'?'selected':''}>timoshenko</option><option ${s.massform==='lumped'?'selected':''}>lumped</option></select></label>`;
    $('input[type=checkbox]',r2).onchange=e=>s.shear=e.target.checked;
    $('select',r2).onchange=e=>s.massform=e.target.value;
    c.appendChild(r2);
    box.appendChild(c);
  });
}
$('#addSec').onclick=()=>{ addSection(); renderSections(); };

/* ===================== CONSTRAINTS ===================== */
function renderConstraints(){
  const box=$('#conList'); box.innerHTML='';
  M.constraints.forEach(c=>{
    const d=document.createElement('div'); d.className='ex'; d.style.cursor='default';
    d.innerHTML=`<div class="row" style="margin:0 0 6px"><label>Name</label><input class="grow nm" type="text" value="${esc(c.name)}"><button class="delx">×</button></div>
      <div class="row"><label>Master nodes</label><input class="grow mn" type="text" value="${esc(c.masters)}" placeholder="e.g. 1,4 or 1-3 — or click nodes"></div>
      <div class="row"><label>Slave nodes</label><input class="grow sn" type="text" value="${esc(c.slaves)}" placeholder="blank = fix to ground — or click nodes"></div>
      <div class="row"><label>DOFs</label><input class="grow df" type="text" value="${esc(c.dofs)}" placeholder="all · trans · rot · 0,1,2"></div>
      <div class="tip">${c.slaves&&c.slaves.trim()?'Ties master↔slave DOFs (lengths must match).':'Support: constrains listed DOFs of master nodes to ground.'}</div>`;
    $('.nm',d).oninput=e=>c.name=e.target.value;
    $('.mn',d).oninput=e=>{c.masters=e.target.value;scheduleDraw();};
    $('.sn',d).oninput=e=>{c.slaves=e.target.value;renderConstraints();scheduleDraw();};
    $('.df',d).oninput=e=>{c.dofs=e.target.value;scheduleDraw();};
    $('.delx',d).onclick=()=>{ M.constraints=M.constraints.filter(x=>x!==c); renderConstraints(); scheduleDraw(); };
    box.appendChild(d);
  });
}
$('#addCon').onclick=()=>{ M.constraints.push({id:uid(),name:'support'+(M.constraints.length+1),masters:'',slaves:'',dofs:'all'}); renderConstraints(); };

/* ===================== FEATURES ===================== */
function renderFeatures(){
  const box=$('#featList'); box.innerHTML='';
  M.features.forEach(f=>{
    const d=document.createElement('div'); d.className='ex'; d.style.cursor='default';
    const isMat=f.ftype==='Matrix';
    const tyOpts=['Spring','Dashpot','PointMass','Matrix'].map(t=>`<option ${f.ftype===t?'selected':''}>${t}</option>`).join('');
    let body=`<div class="row" style="margin:0 0 6px"><label>Type</label>
        <select class="grow ty">${tyOpts}</select><button class="delx">×</button></div>`;
    if(isMat){
      const mt=f.mtype||'k';
      body+=`<div class="row"><label>Name</label><input class="grow nm" type="text" value="${esc(f.name||'')}" placeholder="optional"></div>
      <div class="row"><label>Matrix type</label><select class="grow mt"><option value="k" ${mt==='k'?'selected':''}>k — stiffness</option><option value="c" ${mt==='c'?'selected':''}>c — damping</option><option value="m" ${mt==='m'?'selected':''}>m — mass</option></select></div>
      <div class="row"><label>Node(s)</label><input class="grow nd" type="text" value="${esc(f.nodes)}" placeholder="1  (or 2,11 to couple two nodes)"></div>
      <div class="row"><label>DOFs</label><input class="grow df" type="text" value="${esc(f.dofs)}" placeholder="0,1,2,3,4,5"></div>
      <textarea class="mx" rows="${Math.min(8,Math.max(3,(f.matrix||[]).length))}" placeholder="one row per line, values separated by commas">${esc(matrixText(f.matrix))}</textarea>
      <div class="tip mxinfo"></div>`;
    } else {
      body+=`<div class="row"><label>Node(s)</label><input class="grow nd" type="text" value="${esc(f.nodes)}" placeholder="1 (or 2,11 to connect) — or click nodes"></div>
      <div class="row"><label>DOFs</label><input class="grow df" type="text" value="${esc(f.dofs)}" placeholder="0,1,2"></div>
      <div class="row"><label>${f.ftype==='Spring'?'k':f.ftype==='Dashpot'?'c':'m'} value</label><input class="grow vl" type="number" step="any" value="${esc(f.value)}"></div>`;
    }
    d.innerHTML=body;
    $('.ty',d).onchange=e=>{ f.ftype=e.target.value;
      if(f.ftype==='Matrix'&&!f.matrix){ const n=Math.max(1,parseInts(f.dofs).length*Math.max(1,parseInts(f.nodes).length)); f.mtype='k'; f.matrix=Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?(+f.value||0):0)); }
      renderFeatures(); scheduleDraw(); };
    $('.nd',d).oninput=e=>{f.nodes=e.target.value;scheduleDraw();if(isMat)chk();};
    $('.df',d).oninput=e=>{f.dofs=e.target.value;if(isMat)chk();};
    const chk=()=>{ const n=parseInts(f.nodes).length*parseInts(f.dofs).length, mat=f.matrix||[];
      const ok=mat.length===n && mat.every(r=>r.length===n && r.every(isFinite));
      $('.mxinfo',d).innerHTML= ok? `${n}×${n} matrix over ${parseInts(f.nodes).length} node(s) × ${parseInts(f.dofs).length} DOF(s).` : `<span style="color:var(--red)">Matrix must be ${n}×${n} (nodes × DOFs); currently ${mat.length}×${mat[0]?mat[0].length:0}.</span>`; };
    if(isMat){
      $('.nm',d).oninput=e=>f.name=e.target.value;
      $('.mt',d).onchange=e=>{f.mtype=e.target.value;scheduleDraw();};
      $('.mx',d).oninput=e=>{f.matrix=parseMatrix(e.target.value);chk();};
      chk();
    } else $('.vl',d).oninput=e=>f.value=num(e.target.value);
    $('.delx',d).onclick=()=>{M.features=M.features.filter(x=>x!==f);renderFeatures();scheduleDraw();};
    box.appendChild(d);
  });
}
$('#addFeat').onclick=()=>{ M.features.push({id:uid(),ftype:'Spring',nodes:'',dofs:'0,1,2',value:1e6}); renderFeatures(); };

/* ===================== FORCES ===================== */
function renderForces(){
  const box=$('#forceList'); box.innerHTML='';
  M.forces.forEach(f=>{
    const d=document.createElement('div'); d.className='ex'; d.style.cursor='default';
    const dofOpts=DOFNAMES[M.domain].map((n,i)=>`<option value="${i}" ${+f.dof===i?'selected':''}>${i} — ${n}</option>`).join('');
    d.innerHTML=`<div class="row" style="margin:0 0 6px"><label>Name</label><input class="grow nm" type="text" value="${esc(f.name)}"><button class="delx">×</button></div>
      <div class="row"><label>Node(s)</label><input class="grow nd" type="text" value="${esc(f.nodes)}" placeholder="e.g. 13 or 1-8 — or click nodes"></div>
      <div class="row"><label>DOF</label><select class="grow dof">${dofOpts}</select></div>
      <div class="row"><label>Case</label><select class="grow cs"><option value="analysis" ${f.case!=='base'?'selected':''}>analysis</option><option value="base" ${f.case==='base'?'selected':''}>static base (preload)</option></select></div>
      <div class="row"><label>Mode</label><select class="grow md"><option value="constant" ${f.mode==='constant'?'selected':''}>constant (static)</option><option value="ramp" ${f.mode==='ramp'?'selected':''}>ramp 0→amp</option><option value="history" ${f.mode==='history'?'selected':''}>time history a(t)</option></select></div>
      <div class="row amprow"><label>Amplitude</label><input class="grow amp" type="number" step="any" value="${esc(f.amp)}"></div>
      <div class="row exprow" style="${f.mode==='history'?'':'display:none'}"><label>a(t) =</label><input class="grow ex" type="text" value="${esc(f.texpr)}" placeholder="1000e3*sin(2*pi*0.37*t)"></div>
      <div class="tip">Applied to DOF ${f.dof} (${DOFNAMES[M.domain][+f.dof]||''}) of each listed node. ${f.case==='base'?'Static base load: defines the preload for modal / dynamic analysis with a base state (always included in static analyses).':'Static analyses use all loads; modal / dynamic analyses use the “analysis” loads.'}</div>`;
    $('.nm',d).oninput=e=>f.name=e.target.value;
    $('.nd',d).oninput=e=>{f.nodes=e.target.value;scheduleDraw();};
    $('.dof',d).onchange=e=>{f.dof=+e.target.value;renderForces();scheduleDraw();};
    $('.md',d).onchange=e=>{f.mode=e.target.value;renderForces();};
    $('.cs',d).onchange=e=>{f.case=e.target.value;renderForces();scheduleDraw();};
    $('.amp',d).oninput=e=>f.amp=num(e.target.value);
    const ex=$('.ex',d); if(ex) ex.oninput=e=>f.texpr=e.target.value;
    $('.delx',d).onclick=()=>{M.forces=M.forces.filter(x=>x!==f);renderForces();scheduleDraw();};
    box.appendChild(d);
  });
}
$('#addForce').onclick=()=>{ M.forces.push({id:uid(),name:'load'+(M.forces.length+1),nodes:'',dof:(M.domain==='2d'?1:2),amp:1000,mode:'constant',texpr:'1000*sin(2*pi*1*t)'}); renderForces(); scheduleDraw(); };

/* ---- move all loads between the cases 'analysis' and 'base' (static base / preload) ---- */
function setAllLoadCases(target){
  const all=[...M.forces, ...(M.element_loads||[])];
  if(!all.length){ toast('No loads defined','err'); return; }
  let moved=0, skipped=0;
  all.forEach(f=>{
    if(target==='base' && f.mode==='history'){ if(f.case==='base') f.case='analysis'; skipped++; return; }   // a(t) cannot be static
    if((f.case==='base'?'base':'analysis')!==target){ f.case=target; moved++; }
  });
  renderForces(); renderElLoads(); renderAnalysisParams(); scheduleDraw(); hideResult();
  let msg= target==='base'? `${moved} load(s) moved to static base` : `${moved} load(s) moved to analysis`;
  if(skipped) msg+=` · ${skipped} time-history load(s) kept as analysis loads`;
  const a=M.analysis;
  if(target==='base' && !['eig','lin_dynamic','dynamic'].includes(a.kind)) msg+=' · note: static analyses use all loads';
  else if(target==='base' && (a.base||'none')==='none') msg+=' · choose a static base state in the Analysis tab';
  toast(msg,'ok');
}
$('#allToBase').onclick=()=>setAllLoadCases('base');
$('#allToAnalysis').onclick=()=>setAllLoadCases('analysis');

/* ===================== ELEMENT LOADS ===================== */
const ELOPT={gravity:'self weight (section mass x g)', mass_per_length:'constant mass per length x g', total_mass:'total mass x g (spread per length)', fun:'expression in sec (load per length)'};
function newElLoad(o={}){ return Object.assign({id:uid(), name:'element load '+(M.element_loads.length+1), target:'all', elements:'', section:'',
  option:'gravity', value:0, fun:'sec.rho*sec.A*9.81', g:9.81, direction:'', include_moments:true, mode:'constant', case:'analysis'}, o); }
function elLoadElements(e){   // GUI-side resolution of the loaded elements (for drawing and counts)
  if(e.target==='list'){ const s=new Set(parseInts(e.elements)); return M.elements.filter(el=>s.has(Number(el.label))); }
  if(e.target==='section') return M.elements.filter(el=>el.section===e.section);
  return M.elements;
}
function renderElLoads(){
  const box=$('#elLoadList'); box.innerHTML='';
  if(!M.element_loads) M.element_loads=[];
  M.element_loads.forEach(e=>{
    const d=document.createElement('div'); d.className='ex'; d.style.cursor='default';
    const secOpts=M.sections.map(s=>`<option ${s.name===e.section?'selected':''}>${esc(s.name)}</option>`).join('');
    const dirPh=M.domain==='2d'?'0,-1 (default: gravity -y)':'0,0,-1 (default: gravity -z)';
    d.innerHTML=`<div class="row" style="margin:0 0 6px"><label>Name</label><input class="grow nm" type="text" value="${esc(e.name)}"><button class="delx">×</button></div>
      <div class="row"><label>Elements</label><select class="grow tg"><option value="all" ${e.target==='all'?'selected':''}>all elements</option><option value="list" ${e.target==='list'?'selected':''}>list</option><option value="section" ${e.target==='section'?'selected':''}>all with section</option></select></div>
      ${e.target==='list'?`<div class="row"><label>Element list</label><input class="grow els" type="text" value="${esc(e.elements)}" placeholder="e.g. 1-20,55"><button class="btn sm ghost fromsel" title="Use the elements selected in the view / selection panel">Use selection</button></div>`:''}
      ${e.target==='section'?`<div class="row"><label>Section</label><select class="grow sc">${secOpts}</select></div>`:''}
      <div class="row"><label>Intensity</label><select class="grow op">${Object.entries(ELOPT).map(([k,v])=>`<option value="${k}" ${e.option===k?'selected':''}>${v}</option>`).join('')}</select></div>
      ${e.option==='mass_per_length'||e.option==='total_mass'?`<div class="row"><label>${e.option==='total_mass'?'Total mass':'Mass / length'}</label><input class="grow vl" type="number" step="any" value="${esc(e.value)}"></div>`:''}
      ${e.option==='fun'?`<div class="row"><label>q(sec) =</label><input class="grow fn" type="text" value="${esc(e.fun)}" placeholder="sec.rho*sec.A*9.81"></div>`:`<div class="row"><label>g</label><input class="grow gg" type="number" step="any" value="${esc(e.g)}"></div>`}
      <div class="row"><label>Direction</label><input class="grow dr" type="text" value="${esc(e.direction||'')}" placeholder="${dirPh}"></div>
      <div class="row"><label>Mode</label><select class="grow md"><option value="constant" ${e.mode!=='ramp'?'selected':''}>constant</option><option value="ramp" ${e.mode==='ramp'?'selected':''}>ramp 0 to full over t</option></select></div>
      <div class="row"><label>Case</label><select class="grow cs"><option value="analysis" ${e.case!=='base'?'selected':''}>analysis</option><option value="base" ${e.case==='base'?'selected':''}>static base (preload)</option></select></div>
      <div class="row"><label class="inline"><input type="checkbox" class="im" ${e.include_moments!==false?'checked':''}> consistent end moments (qL²/12)</label></div>
      <div class="tip elinfo"></div>`;
    const info=()=>{ const n=elLoadElements(e).length; $('.elinfo',d).textContent=`${n} element(s). `+(e.option==='gravity'?'Uses the mass per length of each section (rho*A, or m if given).':''); };
    $('.nm',d).oninput=ev=>{e.name=ev.target.value;};
    $('.tg',d).onchange=ev=>{e.target=ev.target.value; if(e.target==='section'&&!e.section&&M.sections[0]) e.section=M.sections[0].name; renderElLoads(); scheduleDraw();};
    const els=$('.els',d); if(els) els.oninput=ev=>{e.elements=ev.target.value; info(); scheduleDraw();};
    const fs=$('.fromsel',d); if(fs) fs.onclick=()=>{ if(!SEL.size){ toast('Select elements first (view, list or section)','err'); return; } e.elements=compactInts([...SEL].sort((a,b)=>a-b)); renderElLoads(); scheduleDraw(); };
    const sc=$('.sc',d); if(sc) sc.onchange=ev=>{e.section=ev.target.value; info(); scheduleDraw();};
    $('.op',d).onchange=ev=>{e.option=ev.target.value; renderElLoads(); scheduleDraw();};
    const vl=$('.vl',d); if(vl) vl.oninput=ev=>{e.value=num(ev.target.value);};
    const fn=$('.fn',d); if(fn) fn.oninput=ev=>{e.fun=ev.target.value;};
    const gg=$('.gg',d); if(gg) gg.oninput=ev=>{e.g=num(ev.target.value,9.81);};
    $('.dr',d).oninput=ev=>{e.direction=ev.target.value; scheduleDraw();};
    $('.md',d).onchange=ev=>{e.mode=ev.target.value;};
    $('.cs',d).onchange=ev=>{e.case=ev.target.value; scheduleDraw();};
    $('.im',d).onchange=ev=>{e.include_moments=ev.target.checked;};
    $('.delx',d).onclick=()=>{ M.element_loads=M.element_loads.filter(x=>x!==e); renderElLoads(); scheduleDraw(); };
    info(); box.appendChild(d);
  });
}
$('#addElLoad').onclick=()=>{ M.element_loads.push(newElLoad()); renderElLoads(); scheduleDraw(); };
$('#addSelfWeight').onclick=()=>{ M.element_loads.push(newElLoad({name:'self weight'})); renderElLoads(); scheduleDraw(); };
/* global direction of an element load (default gravity) */
function elLoadDir(e){
  const v=String(e.direction||'').split(/[\s,;]+/).filter(Boolean).map(Number);
  const d= M.domain==='2d'? (v.length>=2&&v.every(isFinite)&&(v[0]||v[1])? [v[0],v[1],0] : [0,-1,0])
                          : (v.length>=3&&v.every(isFinite)&&(v[0]||v[1]||v[2])? v.slice(0,3) : [0,0,-1]);
  return normalize(d);
}
/* distributed-load arrows along loaded elements (dashed for static base loads) */
function drawElementLoads(){
  if(!M.element_loads||!M.element_loads.length) return;
  const bb=modelBBox(); ctx.font='10.5px ui-monospace,Consolas,monospace';
  M.element_loads.forEach(e=>{
    const els=elLoadElements(e).filter(elVisible); if(!els.length) return;
    const dir=elLoadDir(e), many=els.length>300, isBase=e.case==='base';
    const Ld=bb.size*0.035;
    ctx.strokeStyle=COL.load; ctx.fillStyle=COL.load; ctx.lineWidth=1.2; ctx.globalAlpha=0.85;
    let labelAt=null;
    els.forEach(el=>{ const n1=findNode(el.n1),n2=findNode(el.n2); if(!n1||!n2) return;
      const a=nodeW(n1), b=nodeW(n2);
      const ts= many? [0.5] : [0.15,0.5,0.85];
      const tails=[];
      ts.forEach(t=>{ const p=[a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
        const head=project(p), tail=project(add3(p,dir,-Ld)); tails.push(tail);
        const dx=head.x-tail.x, dy=head.y-tail.y, len=Math.hypot(dx,dy);
        if(len<3) return;
        if(isBase) ctx.setLineDash([3,2]);
        ctx.beginPath(); ctx.moveTo(tail.x,tail.y); ctx.lineTo(head.x-dx/len*5,head.y-dy/len*5); ctx.stroke(); ctx.setLineDash([]);
        arrowHead(head.x,head.y,dx,dy,6); });
      if(!many && tails.length>1){ ctx.beginPath(); ctx.moveTo(tails[0].x,tails[0].y); tails.slice(1).forEach(p=>ctx.lineTo(p.x,p.y)); ctx.stroke(); }
      if(!labelAt) labelAt=tails[Math.floor(tails.length/2)];
    });
    ctx.globalAlpha=1;
    if(labelAt){ const q= e.option==='gravity'? `g=${fmtNum(+e.g||9.81)}` : e.option==='fun'? `q=${e.fun}` : e.option==='total_mass'? `M=${fmtNum(+e.value)}` : `m=${fmtNum(+e.value)}`;
      ctx.fillText(`${e.name}: ${q}${isBase?' (base)':''}`, labelAt.x+6, labelAt.y-6); }
  });
}

function esc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function refreshNodeRefs(){ /* placeholder for future label remap */ }

/* ===================== ANALYSIS PARAMS ===================== */
const ANA_INFO = {
  lin_static:"Linear static. Solves K·u = f once (forces at final time). Fast; ignores geometric nonlinearity.",
  static:"Nonlinear static, corotational Newton–Raphson. Forces ramp over increments (dt<1 ⇒ multiple steps). Set tolerances for convergence.",
  eig:"Eigenvalue analysis: natural frequencies & mode shapes (beef run_eig, state-space, includes damping). Loads are ignored. With solver 'auto', large models are solved for undamped modes with the sparse solver. Blank 'Modes to list' = 30 (20 for sparse).",
  lin_dynamic:"Linear Newmark time integration. Needs a time axis (t0, dt, tmax) and ideally history loads. Rayleigh damping supported.",
  dynamic:"Nonlinear Newmark time integration (corotational). Same time settings; iterates each step to tolerance.",
  buckling:"Linear buckling. Applies the reference load, forms the geometric stiffness, and solves for critical load factors λ (buckling load = λ × reference load)."
};
function renderAnalysisParams(){
  const a=M.analysis, box=$('#anaParams'); box.innerHTML='';
  const kind=a.kind;
  const add=(html)=>{ const w=document.createElement('div'); w.innerHTML=html; box.appendChild(w); return w; };
  const timeRow = ()=>`
    <div class="row"><label>t₀ / dt / tmax</label>
      <input type="number" step="any" id="pa_t0" value="${a.t0}" style="width:60px" title="start time">
      <input type="number" step="any" id="pa_dt" value="${a.dt}" style="width:60px" title="time step">
      <input type="number" step="any" id="pa_tmax" value="${a.tmax}" style="width:60px" title="end time"></div>`;
  const tolRow = ()=>`
    <div class="row"><label>itmax / tol u,r</label>
      <input type="number" id="pa_itmax" value="${a.itmax}" style="width:50px">
      <input type="number" step="any" id="pa_tolU" value="${a.tolU}" placeholder="u" style="width:60px">
      <input type="number" step="any" id="pa_tolR" value="${a.tolR}" placeholder="r" style="width:60px"></div>`;
  const rayRow = ()=>`
    <div class="row"><label>Rayleigh α_m,α_k</label>
      <input type="number" step="any" id="pa_rm" value="${a.rMass}" style="width:74px" title="mass-proportional">
      <input type="number" step="any" id="pa_rk" value="${a.rStiff}" style="width:74px" title="stiffness-proportional"></div>`;
  const nmRow = ()=>`
    <div class="row"><label>Newmark β,γ,α</label>
      <input type="number" step="any" id="pa_beta" value="${a.beta}" style="width:56px">
      <input type="number" step="any" id="pa_gamma" value="${a.gamma}" style="width:56px">
      <input type="number" step="any" id="pa_alpha" value="${a.alpha}" style="width:56px" title="HHT alpha"></div>`;

  const baseRow = ()=>`
    <h3 class="sect" style="margin-top:10px">Static base state</h3>
    <div class="row"><label>Base</label><select class="grow" id="pa_base">
      <option value="none" ${(a.base||'none')==='none'?'selected':''}>none — undeformed, unstressed</option>
      <option value="lin_static" ${a.base==='lin_static'?'selected':''}>linear static: K + K_g(N₀)</option>
      <option value="static" ${a.base==='static'?'selected':''}>nonlinear static: deformed tangent state</option></select></div>
    ${a.base==='static'?`<div class="row"><label>Base load steps</label><input type="number" id="pa_bsteps" value="${a.baseSteps||10}" class="grow"></div>`:''}
    ${a.base==='static' && kind!=='static' && kind!=='dynamic' ? tolRow() : ''}
    <div class="btnrow"><button class="btn sm ghost" id="pa_allbase">Move all loads to static base</button></div>
    <div class="tip">The base analysis uses the loads with case <b>static base</b>. Linear: undeformed geometry with the geometric stiffness of the base axial forces. Nonlinear: corotational static; the following analysis uses the deformed geometry and tangent stiffness (incl. geometric stiffness).</div>
    ${kind==='dynamic' && (a.base||'none')!=='none' ? `<div class="warn">beef's nonlinear dynamic analysis always starts from the undeformed state, so it cannot follow a base state. Use linear dynamic (linearized about the base state).</div>`:''}`;
  if(kind==='lin_static'){ add(`<div class="tip">No extra parameters needed.</div>`); }
  else if(kind==='static'){ add(timeRow()+tolRow()); }
  else if(kind==='eig'){
    add(`<div class="row"><label>Modes to list</label><input type="number" id="pa_nmodes" value="${a.nModes}" placeholder="all" class="grow"></div>
         <div class="row"><label class="inline"><input type="checkbox" id="pa_kg" ${a.include_kg?'checked':''}> include linear geometric stiffness from element N0</label></div>`+baseRow());
  }
  else if(kind==='lin_dynamic'){ add(timeRow()+rayRow()+nmRow()+baseRow()); }
  else if(kind==='dynamic'){ add(timeRow()+tolRow()+rayRow()+nmRow()+baseRow()); }
  else if(kind==='buckling'){ add(`<div class="row"><label>Modes to list</label><input type="number" id="pa_nmodes" value="${a.nModes}" placeholder="all" class="grow"></div><div class="tip">Define a reference load in the Loads tab.</div>`); }

  const bind=(id,key,isNum=true)=>{ const el=$('#'+id); if(el) el.oninput=e=>{ a[key]= (e.target.type==='checkbox')? e.target.checked : (e.target.value===''?'':(isNum?num(e.target.value):e.target.value)); }; };
  bind('pa_t0','t0');bind('pa_dt','dt');bind('pa_tmax','tmax');bind('pa_itmax','itmax');
  bind('pa_tolU','tolU');bind('pa_tolR','tolR');bind('pa_rm','rMass');bind('pa_rk','rStiff');
  bind('pa_beta','beta');bind('pa_gamma','gamma');bind('pa_alpha','alpha');bind('pa_nmodes','nModes');
  const kg=$('#pa_kg'); if(kg) kg.onchange=e=>a.include_kg=e.target.checked;
  const pb=$('#pa_base'); if(pb) pb.onchange=e=>{ a.base=e.target.value; renderAnalysisParams(); hideResult(); };
  const pab=$('#pa_allbase'); if(pab) pab.onclick=()=>setAllLoadCases('base');
  bind('pa_bsteps','baseSteps');
  $('#anaNote').textContent=ANA_INFO[kind]||'';
}
$('#anaKind').onchange=e=>{ M.analysis.kind=e.target.value; renderAnalysisParams(); hideResult(); };

/* ===================== LINE GENERATOR ===================== */
function openLineGen(){
  const html=`<div class="modal show" id="lineGen"><div class="box" style="max-width:460px">
    <button class="closex" data-closelg>×</button><h2>Generate line of nodes</h2>
    <p>Creates evenly spaced nodes between two points, and (optionally) beam elements chaining them.</p>
    <div class="row"><label>Start x,y,z</label><input id="lg_x0" value="0" style="width:52px"><input id="lg_y0" value="0" style="width:52px"><input id="lg_z0" value="0" style="width:52px" ${M.domain==='2d'?'disabled':''}></div>
    <div class="row"><label>End x,y,z</label><input id="lg_x1" value="10" style="width:52px"><input id="lg_y1" value="0" style="width:52px"><input id="lg_z1" value="0" style="width:52px" ${M.domain==='2d'?'disabled':''}></div>
    <div class="row"><label># elements</label><input id="lg_n" type="number" value="10" class="grow"></div>
    <div class="row"><label>First node label</label><input id="lg_lbl" type="number" value="${(M.nodes.reduce((m,n)=>Math.max(m,n.label),0)||0)+1}" class="grow"></div>
    <div class="row"><label class="inline"><input type="checkbox" id="lg_el" checked> also create elements</label></div>
    <div class="btnrow"><button class="btn pri" id="lg_go">Generate</button></div>
  </div></div>`;
  const wrap=document.createElement('div'); wrap.innerHTML=html; document.body.appendChild(wrap.firstChild);
  const close=()=>$('#lineGen').remove();
  $('[data-closelg]').onclick=close;
  $('#lg_go').onclick=()=>{
    const n=Math.max(1,parseInt($('#lg_n').value)||1);
    const p0=[num($('#lg_x0').value),num($('#lg_y0').value),M.domain==='2d'?0:num($('#lg_z0').value)];
    const p1=[num($('#lg_x1').value),num($('#lg_y1').value),M.domain==='2d'?0:num($('#lg_z1').value)];
    let lbl=parseInt($('#lg_lbl').value)||1; const mkEl=$('#lg_el').checked;
    if(mkEl && !M.sections.length) addSection();
    const first=lbl, labels=[];
    for(let i=0;i<=n;i++){ const t=i/n; addNode(p0[0]+(p1[0]-p0[0])*t,p0[1]+(p1[1]-p0[1])*t,p0[2]+(p1[2]-p0[2])*t,lbl); labels.push(lbl); lbl++; }
    if(mkEl) for(let i=0;i<n;i++) addEl(labels[i],labels[i+1]);
    close(); redrawAll(); toast('Generated '+(n+1)+' nodes','ok');
  };
}

/* ===================== SAVE / LOAD ===================== */
$('#btnSave').onclick=()=>{
  const name=(M.name||'model').replace(/[^\w\-]+/g,'_');
  const out=Object.assign({format:'beef-gui-model',version:1,name:M.name||'model',description:M.description||''}, modelPayload());
  saveFile(name+'.beef', JSON.stringify(out,null,1), 'application/json');
};
$('#btnLoad').onclick=()=>{
  const inp=document.createElement('input'); inp.type='file'; inp.accept='.beef,.json,application/json';
  inp.onchange=()=>{ const f=inp.files[0]; if(!f) return; const r=new FileReader();
    r.onload=()=>{ try{ const d=JSON.parse(r.result); loadModel(d); toast('Model loaded','ok'); }catch(e){ toast('Invalid file','err'); } };
    r.readAsText(f); };
  inp.click();
};
function loadModel(d){
  M=Object.assign(blankModel(d.domain||'3d'),d);
  M.analysis=Object.assign(blankModel().analysis, d.analysis||{});
  // ensure ids
  if(!M.element_loads) M.element_loads=[];
  ['nodes','elements','sections','constraints','features','forces','element_loads'].forEach(k=>{ (M[k]||[]).forEach(o=>{if(!o.id)o.id=uid();}); });
  SHOWALL={nodes:false, els:false}; PICK=null; SEL=new Set(); ISO=null; VISNODES=null; $('#selList').value='';
  REFINE_UNDO=null; $('#undoRefine').style.display='none';
  // labels clutter (and slow) the view for big models — start with them off
  VIEW.labels = M.nodes.length<=300; $('#tLabels').classList.toggle('on',VIEW.labels);
  $('#domain').value=M.domain; $('#anaKind').value=M.analysis.kind;
  if(ANIM.on) toggleAnim();
  RESULT=null; $('#tResult').disabled=true; clearPickTarget();
  $('#modelName').textContent=M.name? '— '+M.name : '';
  redrawAll(); renderResultsIdle(); updateLegend();
}

/* ===================== ORCHESTRATION ===================== */
let drawPending=false;
function scheduleDraw(){ if(drawPending)return; drawPending=true; requestAnimationFrame(()=>{drawPending=false; draw();}); }
function redrawAll(){
  renderNodes(); renderEls(); renderSections(); renderConstraints(); renderFeatures(); renderForces(); renderElLoads(); renderAnalysisParams(); renderSelPanel();
  scheduleDraw();
}
$('#domain').onchange=e=>{
  M.domain=e.target.value;
  // fix default force dof if out of range
  M.forces.forEach(f=>{ if(+f.dof>=ndofs()) f.dof = M.domain==='2d'?2:5; });
  redrawAll();
};

/* ===================== VIEWPORT ===================== */
const cv=$('#view'); const ctx=cv.getContext('2d');
let DPR=Math.max(1,window.devicePixelRatio||1);
/* Orthographic camera: world point `center` is drawn at the screen centre + (panx, pany) pixels.
   Orbit rotates about the point at the screen centre; the wheel zooms about the screen centre. */
let CAM={az:-0.9, el:0.5, zoom:40, panx:0, pany:0, center:[0,0,0]};
let VIEW={labels:true, supports:true, loads:true, axes:true, deformed:false, diagrams:false, result:true, eaxes:false, csys:false};
let PICK=null;          // {type:'node'|'element', id}
let PICKTARGET=null;    // focused node-list input that receives clicked nodes
let ANIM={on:false, phase:0, raf:0};
let DIAG='Mz';          // active section-force component
/* colours: structure green, loads blue, deformed orange, supports white, selection cyan-white */
const COL={ member:'#4fbf7a', bar:'#5f8f6e', deformed:'#ff8a3d', node:'#a8d8b9', support:'#eef2f6', tie:'#c58bff',
  load:'#4aa3ff', moment:'#7fd3ff', spring:'#36c2a6', dashpot:'#b48cff', mass:'#ffcf5c', diag:'#ffcf5c', sel:'#e8fbff' };

/* ---- element selection & visibility ---- */
let SEL=new Set();      // selected element labels
let ISO=null;           // null = all visible, otherwise Set of visible element labels
let VISNODES=null;      // cache: Set of node labels on visible elements (null = all)
function elVisible(el){ return !ISO || ISO.has(Number(el.label)); }
function visNodes(){
  if(!ISO) return null;
  if(!VISNODES){ VISNODES=new Set(); M.elements.forEach(el=>{ if(elVisible(el)){ VISNODES.add(Number(el.n1)); VISNODES.add(Number(el.n2)); } }); }
  return VISNODES;
}
function nodeVisible(lbl){ const v=visNodes(); return !v || v.has(Number(lbl)); }

function resizeCanvas(){
  const w=cv.clientWidth, h=cv.clientHeight; DPR=Math.max(1,window.devicePixelRatio||1);
  if(w>0&&h>0){ cv.width=Math.round(w*DPR); cv.height=Math.round(h*DPR); }
  ctx.setTransform(DPR,0,0,DPR,0,0);
  scheduleDraw();
}
window.addEventListener('resize',resizeCanvas);

function cross(a,b){return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];}
function normalize(a){const n=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/n,a[1]/n,a[2]/n];}
function sub(a,b){return [a[0]-b[0],a[1]-b[1],a[2]-b[2]];}
function add3(a,b,s=1){return [a[0]+b[0]*s,a[1]+b[1]*s,a[2]+b[2]*s];}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}

function camBasis(){
  if(M.domain==='2d') return {right:[1,0,0], up:[0,1,0], fwd:[0,0,-1]};
  const az=CAM.az, el=CAM.el;
  const f=[Math.cos(el)*Math.cos(az), Math.cos(el)*Math.sin(az), Math.sin(el)];   // from scene towards viewer
  // right-handed screen frame: right × up = f (towards the viewer). Front view: x right, z up; top: x right, y up.
  const zf=cross([0,0,1],f);
  let r= Math.hypot(...zf)<1e-9? [1,0,0] : normalize(zf);
  return {right:r, up:cross(f,r), fwd:f};
}
function project(p){ // world [x,y,z] -> screen {x,y}
  const b=camBasis(), d=sub(p,CAM.center);
  return { x: cv.clientWidth/2 + dot(d,b.right)*CAM.zoom + CAM.panx,
           y: cv.clientHeight/2 - dot(d,b.up)*CAM.zoom + CAM.pany };
}
function nodeW(n){ return [+n.x, +n.y, M.domain==='2d'?0:+(n.z||0)]; }
function modelBBox(){
  if(!M.nodes.length) return {min:[0,0,0],max:[1,1,0],size:1,center:[0,0,0]};
  const mn=[1e30,1e30,1e30],mx=[-1e30,-1e30,-1e30];
  M.nodes.forEach(n=>{ const p=nodeW(n); for(let i=0;i<3;i++){mn[i]=Math.min(mn[i],p[i]);mx[i]=Math.max(mx[i],p[i]);} });
  const size=Math.max(mx[0]-mn[0],mx[1]-mn[1],mx[2]-mn[2],1e-6);
  return {min:mn,max:mx,size,center:[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2]};
}
/* Fit: frame the projected extent of all nodes (not the bounding box) in the current view direction */
/* world points to frame: the visible nodes, and the deformed / mode shape at the current scale when it is drawn
   (both extremes while animating), so Fit and the view buttons centre what is actually on screen */
function framedPoints(){
  const nodes=M.nodes.filter(n=>nodeVisible(n.label));
  const pts=nodes.map(nodeW);
  const showDef = resultShown() && (VIEW.deformed||ANIM.on||isModal());
  const disp = showDef? getNodeDisp() : null;
  if(disp){
    computeAutoBase(disp);
    const sc=defScale();
    BASEDRAW = (isModal() && baseShown())? {map:baseMap(), sc:num($('#baseScale').value,1)} : null;
    nodes.forEach(n=>{ pts.push(dispOf(n,disp,sc)); if(ANIM.on) pts.push(dispOf(n,disp,-sc)); if(BASEDRAW) pts.push(basePos(n)); });
  }
  return pts;
}
function fitView(){
  resizeCanvas();
  const W=cv.clientWidth, H=cv.clientHeight;
  if(!(W>0&&H>0)){ requestAnimationFrame(fitView); return; }
  const bb=modelBBox(); CAM.center=bb.center; CAM.panx=0; CAM.pany=0;
  const b=camBasis(); let x0=1e30,x1=-1e30,y0=1e30,y1=-1e30;
  const pts=framedPoints();                                      // what is drawn: visible nodes (+ deformed shape)
  pts.forEach(p=>{ const d=sub(p,CAM.center); const sx=dot(d,b.right), sy=dot(d,b.up);
    x0=Math.min(x0,sx); x1=Math.max(x1,sx); y0=Math.min(y0,sy); y1=Math.max(y1,sy); });
  if(!pts.length){ CAM.zoom=40; scheduleDraw(); return; }
  // move the centre to the middle of the projected extent, then zoom to fill 85 %
  CAM.center=add3(add3(CAM.center,b.right,(x0+x1)/2),b.up,(y0+y1)/2);
  const ex=Math.max(x1-x0,bb.size*1e-3), ey=Math.max(y1-y0,bb.size*1e-3);
  CAM.zoom=Math.min(0.85*W/ex, 0.85*H/ey);
  scheduleDraw();
}
/* fold screen pan into the camera centre so orbiting pivots about what is in the middle of the screen */
function absorbPan(){
  if(!CAM.panx&&!CAM.pany) return;
  const b=camBasis();
  CAM.center=add3(add3(CAM.center,b.right,-CAM.panx/CAM.zoom),b.up,CAM.pany/CAM.zoom);
  CAM.panx=0; CAM.pany=0;
}

/* ---- displacement field for the current result view ---- */
let AUTOBASE=1;
function resultShown(){ return RESULT && VIEW.result; }
function isModal(){ return RESULT && (RESULT.kind==='eig'||RESULT.kind==='buckling'); }
function getNodeDisp(){ // {label:[dx,dy,dz]} or null
  if(!resultShown()) return null;
  let vec=null;
  if(isModal()){ if(baseView()==='base') return baseMap();          // base state only: the static deformation
    vec=RESULT.phi? RESULT.phi[RESULT.selMode||0] : null; }
  else if(RESULT.u){ const ti=RESULT.timeIndex!=null?RESULT.timeIndex:RESULT.u[0].length-1; vec=RESULT.u.map(row=>row[ti]);
    // time results after a base state are base + increment (same physical scale); unticking "include base state"
    // shows the increment only
    const bv=baseView();
    if(bv==='mode'){ const ub=RESULT.base.u; vec=vec.map((v,i)=>v-(ub[i]||0)); }      // increment only
    else if(bv==='base') vec=RESULT.base.u.slice(); }                                   // static base only
  if(!vec) return null;
  const out={};
  RESULT.nodeLabels.forEach((lbl,i)=>{ const gd=RESULT.node_gdofs[i];
    out[lbl]= RESULT.domain==='2d'? [vec[gd[0]]||0, vec[gd[1]]||0, 0] : [vec[gd[0]]||0, vec[gd[1]]||0, vec[gd[2]]||0]; });
  return out;
}
/* ---- static base state in the view ---- */
let BASEDRAW=null;      // {map, sc} while drawing a modal result about a base state
/* 'both' (base + mode / total), 'base' (static base only) or 'none'-equivalent 'mode' (mode / increment only);
   'both' when the result has no base state */
function baseView(){ return (RESULT && RESULT.base)? ($('#baseView').value||'both') : 'none'; }
function baseShown(){ return baseView()==='both'; }
function baseMap(){
  if(RESULT._baseMap) return RESULT._baseMap;
  const ub=RESULT.base.u, out={};
  RESULT.nodeLabels.forEach((lbl,i)=>{ const gd=RESULT.node_gdofs[i];
    out[lbl]= RESULT.domain==='2d'? [ub[gd[0]]||0, ub[gd[1]]||0, 0] : [ub[gd[0]]||0, ub[gd[1]]||0, ub[gd[2]]||0]; });
  return (RESULT._baseMap=out);
}
function basePos(n){ const p=nodeW(n); const d=BASEDRAW&&BASEDRAW.map[n.label]; const s=BASEDRAW?BASEDRAW.sc:0;
  return d? [p[0]+d[0]*s,p[1]+d[1]*s,p[2]+d[2]*s] : p; }
function drawBaseConfig(){   // deformed base state: the reference the modes oscillate about
  ctx.globalAlpha=0.75; ctx.lineWidth=1.4; ctx.strokeStyle=COL.member; ctx.setLineDash([5,3]);
  ctx.beginPath();
  M.elements.forEach(el=>{ if(!elVisible(el)) return; const n1=findNode(el.n1), n2=findNode(el.n2); if(!n1||!n2) return;
    const a=project(basePos(n1)), b=project(basePos(n2)); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); });
  ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha=1;
}
function computeAutoBase(disp){
  const bb=modelBBox(); let mx=0;
  for(const k in disp){ mx=Math.max(mx, Math.hypot(...disp[k])); }
  AUTOBASE = mx>1e-20 ? (bb.size*0.1)/mx : 1;
}
function defScale(){
  const v=parseFloat($('#scaleRange').value)||0;
  return ($('#autoScale').checked?AUTOBASE:1)*Math.pow(10,v);
}

/* ---- main draw ---- */
function draw(){
  const W=cv.clientWidth,H=cv.clientHeight; VISNODES=null;
  ctx.clearRect(0,0,W,H);
  if(!M.nodes.length){ drawEmpty(W,H); drawAxisGizmo(W,H); return; }
  const showDef = resultShown() && (VIEW.deformed||ANIM.on||isModal());
  const disp=showDef?getNodeDisp():null;
  if(disp) computeAutoBase(disp);
  let sc=disp?defScale():0;
  if(ANIM.on && baseView()!=='base') sc*=Math.sin(ANIM.phase);      // the static base state alone is not animated
  // modal result about a base state: modes are drawn about the deformed base configuration
  BASEDRAW = (disp && isModal() && baseShown())? {map:baseMap(), sc:num($('#baseScale').value,1)} : null;
  drawMembers(false,null,0, disp? 0.35 : 1);
  if(BASEDRAW) drawBaseConfig();
  if(disp) drawMembers(true,disp,sc,1);
  if(VIEW.diagrams && resultShown() && RESULT.elements) drawDiagrams();
  if(VIEW.eaxes) drawElementAxes();
  if(VIEW.csys) drawGlobalCsys();
  if(VIEW.supports) drawSupports();
  drawNodes(disp,sc);
  if(VIEW.loads && !(resultShown()&&isModal())){ drawElementLoads(); drawLoads(); }   // after the nodes: arrow tips stay visible
  if(VIEW.labels) drawLabels();
  drawPickHighlight();
  drawAxisGizmo(W,H);
}
function drawEmpty(W,H){ ctx.fillStyle='#4a586b'; ctx.font='13px system-ui'; ctx.textAlign='center';
  ctx.fillText('No nodes yet — add nodes in the Model tab, or load an Example.',W/2,H/2); ctx.textAlign='left'; }
function dispOf(n,disp,sc){ const p=basePos(n); const d=disp&&disp[n.label]; return d? [p[0]+d[0]*sc,p[1]+d[1]*sc,p[2]+d[2]*sc] : p; }

function drawMembers(def,disp,sc,alpha){
  ctx.globalAlpha=alpha; ctx.lineCap='round';
  const selected=[];
  M.elements.forEach(el=>{
    if(!elVisible(el)) return;
    const n1=findNode(el.n1), n2=findNode(el.n2); if(!n1||!n2) return;
    const a=project(def?dispOf(n1,disp,sc):nodeW(n1)), b=project(def?dispOf(n2,disp,sc):nodeW(n2));
    if(!def && SEL.has(Number(el.label))){ selected.push([a,b]); return; }
    ctx.lineWidth=def?2.2:1.5;
    ctx.strokeStyle= def? COL.deformed : (el.etype==='bar'?COL.bar:COL.member);
    ctx.setLineDash(!def && el.etype==='bar'? [5,4] : []);
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  });
  ctx.setLineDash([]);
  if(selected.length){   // selected elements on top: bright, thicker
    ctx.globalAlpha=Math.max(alpha,0.8); ctx.strokeStyle=COL.sel; ctx.lineWidth=2.8;
    ctx.beginPath(); selected.forEach(([a,b])=>{ ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); }); ctx.stroke();
  }
  ctx.globalAlpha=1;
}
function drawNodes(disp,sc){
  const small=M.nodes.length>500;
  M.nodes.forEach(n=>{
    if(!nodeVisible(n.label)) return;
    const p=project(disp? dispOf(n,disp,sc) : nodeW(n)); ctx.fillStyle=disp?'#ffb27a':COL.node;
    if(small){ ctx.fillRect(p.x-1,p.y-1,2,2); } else { ctx.beginPath(); ctx.arc(p.x,p.y,2.6,0,7); ctx.fill(); }
  });
}
function drawLabels(){
  ctx.font='10px ui-monospace,Consolas,monospace'; ctx.textAlign='left';
  ctx.fillStyle='#b9c6d6';
  M.nodes.forEach(n=>{ if(!nodeVisible(n.label)) return; const p=project(nodeW(n)); ctx.fillText(n.label, p.x+6,p.y-5); });
  ctx.fillStyle='#6f829a';
  M.elements.forEach(el=>{ if(!elVisible(el)) return; const n1=findNode(el.n1),n2=findNode(el.n2); if(!n1||!n2)return;
    const a=nodeW(n1),b=nodeW(n2),mid=project([(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2]);
    ctx.fillText('['+el.label+']', mid.x+4, mid.y+10); });
}
/* label -> node lookup; indexed so drawing large models stays O(elements) */
let NODEIDX=null, NODEIDX_SRC=null, NODEIDX_LEN=-1;
function rebuildNodeIdx(){ NODEIDX=new Map(); M.nodes.forEach(n=>NODEIDX.set(Number(n.label),n)); NODEIDX_SRC=M.nodes; NODEIDX_LEN=M.nodes.length; }
function findNode(lbl){
  if(NODEIDX_SRC!==M.nodes || NODEIDX_LEN!==M.nodes.length) rebuildNodeIdx();
  let n=NODEIDX.get(Number(lbl));
  if(n && n.label==lbl) return n;
  rebuildNodeIdx();
  n=NODEIDX.get(Number(lbl));
  return (n && n.label==lbl)? n : undefined;
}

function supportNodes(){ const s=new Set(); M.constraints.forEach(c=>{ if(!(c.slaves&&String(c.slaves).trim())) parseInts(c.masters).forEach(l=>s.add(l)); }); return s; }
function drawSupports(){
  ctx.strokeStyle=COL.support; ctx.fillStyle='rgba(238,242,246,.14)'; ctx.lineWidth=1.5;
  supportNodes().forEach(lbl=>{ if(!nodeVisible(lbl)) return; const n=findNode(lbl); if(!n)return; const p=project(nodeW(n)); const s=7;
    ctx.beginPath(); ctx.moveTo(p.x,p.y); ctx.lineTo(p.x-s,p.y+s*1.3); ctx.lineTo(p.x+s,p.y+s*1.3); ctx.closePath(); ctx.fill(); ctx.stroke();
  });
  ctx.strokeStyle=COL.tie; ctx.setLineDash([3,3]); ctx.lineWidth=1.2;
  M.constraints.forEach(c=>{ if(c.slaves&&String(c.slaves).trim()){ const m=parseInts(c.masters),s=parseInts(c.slaves);
    for(let i=0;i<Math.min(m.length,s.length);i++){ if(!nodeVisible(m[i])&&!nodeVisible(s[i])) continue; const a=findNode(m[i]),b=findNode(s[i]); if(a&&b){const pa=project(nodeW(a)),pb=project(nodeW(b));ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();}}
  }});
  ctx.setLineDash([]);
  M.features.forEach(f=>{ const ns=parseInts(f.nodes);
    const kind=f.ftype==='Matrix'? ({k:'Spring',c:'Dashpot',m:'PointMass'})[f.mtype||'k'] : f.ftype;
    const col=kind==='PointMass'?COL.mass:kind==='Dashpot'?COL.dashpot:COL.spring;
    ns.forEach(l=>{if(!nodeVisible(l))return;const n=findNode(l);if(!n)return;const p=project(nodeW(n));ctx.strokeStyle=col;ctx.fillStyle=col;ctx.lineWidth=1.4;
      if(f.ftype==='Matrix'){ const s=kind==='PointMass'?6:9; ctx.beginPath();ctx.moveTo(p.x,p.y-s);ctx.lineTo(p.x+s,p.y);ctx.lineTo(p.x,p.y+s);ctx.lineTo(p.x-s,p.y);ctx.closePath();
        if(kind==='PointMass') ctx.fill(); else ctx.stroke(); }
      else if(kind==='PointMass'){ctx.fillRect(p.x-4,p.y-4,8,8);}
      else{ctx.beginPath();ctx.arc(p.x,p.y,7,0,7);ctx.stroke();}});
  });
}
/* Loads as true 3D arrows: a world-space segment of fixed model-relative length, projected, so they
   rotate and foreshorten with the model. Moments use the double-headed vector convention. */
function dofDir(dof){
  if(M.domain==='2d') return [[1,0,0],[0,1,0],[0,0,1]][dof]||[0,1,0];
  return [[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]][dof]||[0,0,1];
}
function loadComp(dof){ return (M.domain==='2d'? ['Fx','Fy','Mz'] : ['Fx','Fy','Fz','Mx','My','Mz'])[dof]||('dof'+dof); }
/* plain number, no units: 1.0e6, -2.5e4, 1.25e6, 500, 0.75 */
function fmtNum(v){
  if(v==null||!isFinite(v)) return '—';
  const a=Math.abs(v);
  if(a===0) return '0';
  if(a>=1e3||a<1e-2){
    const e=Math.floor(Math.log10(a)); let mant=v/Math.pow(10,e);
    let m=String(parseFloat(mant.toPrecision(3)));
    if(Math.abs(parseFloat(m))>=10){ m=String(parseFloat((mant/10).toPrecision(3))); return (m.includes('.')?m:m+'.0')+'e'+(e+1); }
    return (m.includes('.')?m:m+'.0')+'e'+e;
  }
  return String(parseFloat(v.toPrecision(4)));
}
function fmtLoad(v){ return fmtNum(v); }
/* Arrow points in the signed direction of the load; its length scales with |amplitude| (forces and moments
   separately); a label gives component and signed value. History loads use full length and "(t)". */
function drawLoads(){
  const bb=modelBBox(), Lw=bb.size*0.07; ctx.lineWidth=2;
  const isRot=f=>(+f.dof) >= (M.domain==='2d'?2:3);
  const maxA={f:0,m:0};
  M.forces.forEach(f=>{ if(f.mode!=='history') maxA[isRot(f)?'m':'f']=Math.max(maxA[isRot(f)?'m':'f'],Math.abs(+f.amp||0)); });
  ctx.font='10.5px ui-monospace,Consolas,monospace';
  M.forces.forEach(f=>{
    const dof=+f.dof, rot=isRot(f), amp=+f.amp||0;
    const sign=(f.mode==='history')?1:(amp>=0?1:-1);
    const dir=dofDir(dof).map(v=>v*sign);
    const mx=maxA[rot?'m':'f'];
    const L= (f.mode==='history'||!mx)? Lw : Lw*(0.35+0.65*Math.abs(amp)/mx);
    const isBase=f.case==='base';
    const text=(f.mode==='history'? `${loadComp(dof)}(t)` : `${loadComp(dof)} = ${fmtLoad(amp)}`)+(isBase?' (base)':'');
    const col=rot?COL.moment:COL.load; ctx.strokeStyle=col; ctx.fillStyle=col;
    const nodes=parseInts(f.nodes).filter(l=>nodeVisible(l));
    nodes.forEach((l,k)=>{ const n=findNode(l); if(!n)return; const pw=nodeW(n);
      const label=(nodes.length<=20||k===0);
      if(M.domain==='2d' && rot){  // in-plane moment: curved arrow, counter-clockwise = positive
        // screen y points down, so a counter-clockwise (positive) moment is an "anticlockwise" canvas arc
        const p=project(pw), r=10+6*L/Lw, s0=0.5, e0= sign>0? s0-5.0 : s0+5.0;
        ctx.beginPath(); ctx.arc(p.x,p.y,r,s0,e0,sign>0); ctx.stroke();
        const tx= sign>0? Math.sin(e0) : -Math.sin(e0), ty= sign>0? -Math.cos(e0) : Math.cos(e0);
        arrowHead(p.x+r*Math.cos(e0), p.y+r*Math.sin(e0), tx, ty, 10);
        if(label) ctx.fillText(text, p.x+r+5, p.y-r);
        return; }
      const tail=project(rot? pw : add3(pw,dir,-L)), head=project(rot? add3(pw,dir,L) : pw);
      const dx=head.x-tail.x, dy=head.y-tail.y, len=Math.hypot(dx,dy);
      if(len<5){ // pointing (almost) along the view direction: ⊙ towards the viewer, ⊗ away
        const p=project(pw); ctx.beginPath(); ctx.arc(p.x,p.y,6,0,7); ctx.stroke();
        if(dot(dir,camBasis().fwd)>0){ ctx.beginPath(); ctx.arc(p.x,p.y,2,0,7); ctx.fill(); }
        else { ctx.beginPath(); ctx.moveTo(p.x-4,p.y-4); ctx.lineTo(p.x+4,p.y+4); ctx.moveTo(p.x+4,p.y-4); ctx.lineTo(p.x-4,p.y+4); ctx.stroke(); }
        if(label) ctx.fillText(text, p.x+9, p.y-8);
        return; }
      // shaft stops at the base of the head(s) so the head reads as a solid triangle
      const hs=headSize(len), nh=rot?2:1, ux=dx/len, uy=dy/len, back=Math.min(len, hs*(rot?1.7:0.85));
      if(isBase) ctx.setLineDash([5,3]);     // static base (preload) loads: dashed shaft
      ctx.beginPath(); ctx.moveTo(tail.x,tail.y); ctx.lineTo(head.x-ux*back,head.y-uy*back); ctx.stroke();
      ctx.setLineDash([]);
      arrowHead(head.x,head.y,dx,dy,hs);
      if(nh===2) arrowHead(head.x-ux*hs*0.85,head.y-uy*hs*0.85,dx,dy,hs);   // moment: double head
      if(label){ // just beyond the free end of the arrow, on the side pointing away from the node
        const far= rot? head : tail, p=project(pw), ux=(far.x-p.x)/len, uy=(far.y-p.y)/len;
        const w=ctx.measureText(text).width;
        ctx.fillText(text, far.x+ux*6-(ux<0? w : 0), far.y+uy*6+(uy>0? 9 : 0)); }
    });
  });
}
/* arrowhead size in px: grows with the drawn arrow length, 8..14 px */
function headSize(len){ return Math.max(8, Math.min(14, len*0.28)); }
/* filled triangular head with its tip at (x,y), pointing along (dx,dy); thin dark outline for contrast */
function arrowHead(x,y,dx,dy,s=10){ const n=Math.hypot(dx,dy)||1; dx/=n;dy/=n; const w=s*0.5;
  ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(x-dx*s-dy*w, y-dy*s+dx*w); ctx.lineTo(x-dx*s+dy*w, y-dy*s-dx*w); ctx.closePath();
  ctx.fill();
  const lw=ctx.lineWidth, ss=ctx.strokeStyle; ctx.lineWidth=0.8; ctx.strokeStyle='rgba(5,10,16,.85)'; ctx.stroke();
  ctx.lineWidth=lw; ctx.strokeStyle=ss; }

function drawPickHighlight(){
  if(!PICK) return;
  ctx.save(); ctx.strokeStyle='#fff';
  if(PICK.type==='node'){ const n=findNode(PICK.id); if(n){const p=project(nodeW(n));ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,7,0,7);ctx.stroke();}}
  else{ const el=M.elements.find(e=>e.id===PICK.id); if(el){const n1=findNode(el.n1),n2=findNode(el.n2);if(n1&&n2){const a=project(nodeW(n1)),b=project(nodeW(n2));ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}}}
  // nodes already in the active pick field
  if(PICKTARGET){ ctx.strokeStyle=COL.load; ctx.lineWidth=2;
    parseInts(PICKTARGET.value).forEach(l=>{ const n=findNode(l); if(n){ const p=project(nodeW(n)); ctx.beginPath(); ctx.arc(p.x,p.y,6,0,7); ctx.stroke(); } }); }
  ctx.restore();
}
function drawAxisGizmo(W,H){
  const ox=W-46, oy=H-40, L=22; const axes=[[[1,0,0],'#ff5d6c','x'],[[0,1,0],'#39d98a','y'],[[0,0,1],'#4aa3ff','z']];
  const b=camBasis(); ctx.lineWidth=2; ctx.font='10px ui-monospace,Consolas,monospace';
  axes.forEach(([d,c,nm])=>{ if(M.domain==='2d'&&nm==='z')return; const vx=dot(d,b.right), vy=dot(d,b.up);
    ctx.strokeStyle=c;ctx.fillStyle=c; ctx.beginPath();ctx.moveTo(ox,oy);ctx.lineTo(ox+vx*L,oy-vy*L);ctx.stroke();
    ctx.fillText(nm, ox+vx*(L+6)-3, oy-vy*(L+6)+3); });
}


/* ---- element local axes and global coordinate system ---- */
const AXCOL=['#ff5d6c','#39d98a','#4aa3ff'];
/* local axes of a GUI element, as beef establishes them: e1 node1->node2; e2 from the section's e2, or beef's
   fallback (global unit vector of the smallest |component| of the element vector); e3 = e1 x e2p, e2 = e3 x e1 */
function elementAxes(el){
  const n1=findNode(el.n1), n2=findNode(el.n2); if(!n1||!n2) return null;
  const a=nodeW(n1), b=nodeW(n2), v=sub(b,a); if(Math.hypot(...v)===0) return null;
  const e1=normalize(v);
  if(M.domain==='2d') return [e1,[-e1[1],e1[0],0],[0,0,1]];
  if(RESULT && RESULT.elements && RESULT.domain==='3d'){   // exact axes reported by beef for the last result
    const E=RESULT.elements; if(!E._idx){ E._idx=new Map(); E.labels.forEach((l,i)=>E._idx.set(l,i)); }
    const i=E._idx.get(Number(el.label)); if(i!==undefined) return [e1,E.e2[i],E.e3[i]];
  }
  const sec=M.sections.find(s=>s.name===el.section);
  let e2p=null;
  if(sec && sec.e2){ const w=String(sec.e2).split(/[\s,;]+/).filter(Boolean).map(Number); if(w.length===3&&w.every(isFinite)&&(w[0]||w[1]||w[2])) e2p=w; }
  if(!e2p){ const av=v.map(Math.abs); let k=0; for(let j=1;j<3;j++) if(av[j]<av[k]) k=j; e2p=[0,0,0]; e2p[k]=1; }
  const e3t=cross(e1,e2p); if(Math.hypot(...e3t)<1e-12) return null;   // e2 parallel to the element (beef gives NaN)
  const e2=normalize(cross(normalize(e3t),e1)), e3=cross(e1,e2);
  return [e1,e2,e3];
}
function drawElementAxes(){
  let els=M.elements.filter(elVisible);
  if(SEL.size) els=els.filter(e=>SEL.has(Number(e.label)));      // with a selection: only the selected elements
  const MAXN=600, stride=Math.max(1,Math.ceil(els.length/MAXN));
  ctx.lineWidth=1.6; ctx.font='9px ui-monospace,Consolas,monospace';
  els.forEach((el,k)=>{ if(k%stride) return;
    const ax=elementAxes(el); if(!ax) return;
    const a=nodeW(findNode(el.n1)), b=nodeW(findNode(el.n2)), mid=[(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2];
    const pm=project(mid), Lpx=Math.hypot(project(a).x-project(b).x, project(a).y-project(b).y);
    const len=Math.max(10,Math.min(28,0.3*Lpx)), w=len/CAM.zoom;      // screen-size triad (10..28 px)
    (M.domain==='2d'? [0,1] : [0,1,2]).forEach(j=>{
      const p=project(add3(mid,ax[j],w)); const dx=p.x-pm.x, dy=p.y-pm.y; if(Math.hypot(dx,dy)<2) return;
      ctx.strokeStyle=AXCOL[j]; ctx.fillStyle=AXCOL[j];
      ctx.beginPath(); ctx.moveTo(pm.x,pm.y); ctx.lineTo(p.x,p.y); ctx.stroke();
      if(stride===1 && els.length<=60) ctx.fillText('xyz'[j], p.x+2, p.y-2);
    });
  });
  if(stride>1) { ctx.fillStyle='#8aa0b6'; ctx.font='11px system-ui'; ctx.fillText(`element axes: every ${stride}. element shown (select elements to see theirs)`, 12, cv.clientHeight-28); }
}
function drawGlobalCsys(){
  const bb=modelBBox(), L=bb.size*0.08, o=[0,0,0], po=project(o);
  ctx.lineWidth=2.2; ctx.font='bold 11px ui-monospace,Consolas,monospace';
  (M.domain==='2d'? [0,1] : [0,1,2]).forEach(j=>{
    const d=[0,0,0]; d[j]=1; const p=project(add3(o,d,L)); const dx=p.x-po.x, dy=p.y-po.y, n=Math.hypot(dx,dy);
    ctx.strokeStyle=AXCOL[j]; ctx.fillStyle=AXCOL[j];
    if(n<4){ ctx.beginPath(); ctx.arc(po.x,po.y,5,0,7); ctx.stroke(); ctx.fillText('XYZ'[j], po.x+7, po.y-7-12*j); return; }
    ctx.beginPath(); ctx.moveTo(po.x,po.y); ctx.lineTo(p.x-dx/n*6,p.y-dy/n*6); ctx.stroke();
    arrowHead(p.x,p.y,dx,dy,9);
    ctx.fillText('XYZ'[j], p.x+dx/n*8-4, p.y+dy/n*8+4);
  });
  ctx.fillStyle='#e7edf5'; ctx.beginPath(); ctx.arc(po.x,po.y,2.5,0,7); ctx.fill();
  ctx.font='10px ui-monospace,Consolas,monospace'; ctx.fillStyle='#8aa0b6'; ctx.fillText('O', po.x-12, po.y+12);
}

/* ---- section-force diagrams, drawn in the element's local plane ---- */
const DIAG_AXIS={Qy:'e2', Mz:'e2', Q:'e2', M:'e2', Qz:'e3', My:'e3'};
function drawDiagrams(){
  const E=RESULT.elements; if(!E||!E.values[DIAG]) return;
  const vals=E.values[DIAG]; let mx=0;   // scale on the visible elements, so isolated parts stay readable
  vals.forEach((v,i)=>{ if(ISO&&!ISO.has(E.labels[i])) return; mx=Math.max(mx,Math.abs(v[0]),Math.abs(v[1])); }); if(mx<1e-20) return;
  const amp=modelBBox().size*0.08/mx, b=camBasis();
  if(!E._idx){ E._idx=new Map(); E.labels.forEach((l,i)=>E._idx.set(l,i)); }
  ctx.strokeStyle=COL.diag; ctx.fillStyle='rgba(255,207,92,.16)'; ctx.lineWidth=1.2;
  M.elements.forEach(el=>{ if(!elVisible(el)) return; const i=E._idx.get(Number(el.label)); if(i===undefined) return;
    const n1=findNode(el.n1),n2=findNode(el.n2); if(!n1||!n2) return;
    const a=nodeW(n1), c=nodeW(n2);
    let ax=DIAG_AXIS[DIAG];
    if(!ax){ // N, Mx: the local axis that shows best on screen
      const s2=Math.hypot(dot(E.e2[i],b.right),dot(E.e2[i],b.up)), s3=Math.hypot(dot(E.e3[i],b.right),dot(E.e3[i],b.up));
      ax= s2>=s3? 'e2':'e3'; }
    const d=E[ax][i], v=vals[i];
    const o1=project(a), o2=project(c), p1=project(add3(a,d,v[0]*amp)), p2=project(add3(c,d,v[1]*amp));
    ctx.beginPath(); ctx.moveTo(o1.x,o1.y); ctx.lineTo(p1.x,p1.y); ctx.lineTo(p2.x,p2.y); ctx.lineTo(o2.x,o2.y); ctx.closePath(); ctx.fill(); ctx.stroke();
  });
}

/* ---- interaction: drag = orbit, shift/right/middle-drag = pan, wheel = zoom, click = pick ---- */
let drag=null;
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('mousedown',e=>{
  const box = e.button===0 && (e.ctrlKey||e.metaKey);                 // ctrl-drag: box selection
  const pan = !box && (e.shiftKey || e.button===1 || e.button===2 || M.domain==='2d');
  if(!pan && !box) absorbPan();
  drag={x:e.clientX,y:e.clientY,pan,box,az:CAM.az,el:CAM.el,panx:CAM.panx,pany:CAM.pany,moved:false};
  cv.classList.add('drag'); e.preventDefault();
});
window.addEventListener('mousemove',e=>{ if(!drag)return; const dx=e.clientX-drag.x, dy=e.clientY-drag.y;
  if(Math.abs(dx)+Math.abs(dy)>3) drag.moved=true;
  if(drag.box){ const r=cv.getBoundingClientRect(), b=$('#selBox');
    b.style.left=(Math.min(drag.x,e.clientX)-r.left)+'px'; b.style.top=(Math.min(drag.y,e.clientY)-r.top)+'px';
    b.style.width=Math.abs(dx)+'px'; b.style.height=Math.abs(dy)+'px'; b.style.display= drag.moved?'block':'none'; return; }
  if(!drag.pan){ CAM.az=drag.az-dx*0.008; CAM.el=Math.max(-1.55,Math.min(1.55,drag.el+dy*0.008)); }
  else { CAM.panx=drag.panx+dx; CAM.pany=drag.pany+dy; }
  scheduleDraw();
});
window.addEventListener('mouseup',e=>{
  if(!drag) return;
  if(drag.box && drag.moved){ const r=cv.getBoundingClientRect();
    boxSelect(Math.min(drag.x,e.clientX)-r.left, Math.min(drag.y,e.clientY)-r.top, Math.max(drag.x,e.clientX)-r.left, Math.max(drag.y,e.clientY)-r.top, e.shiftKey); }
  else if(!drag.moved && e.button===0) pickAt(e);
  $('#selBox').style.display='none'; drag=null; cv.classList.remove('drag');
});
cv.addEventListener('wheel',e=>{ e.preventDefault();
  const f=Math.exp(-(e.deltaY||0)*(e.deltaMode===1?0.05:0.0015));
  CAM.zoom*=f; CAM.panx*=f; CAM.pany*=f;   // pure zoom about the screen centre: nothing slides sideways
  scheduleDraw(); },{passive:false});

function pickAt(e){
  const r=cv.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
  const additive=!!(e.ctrlKey||e.metaKey||e.shiftKey);
  // node picking: generous radius when filling a node field; otherwise only a near-direct hit on a node
  // (so elements stay clickable in dense views), and never for shift/ctrl-click (element selection)
  const picking=PICKTARGET && PICKTARGET.isConnected;
  let best=null, bd= picking? 12 : (additive? -1 : 6);
  M.nodes.forEach(n=>{ if(!nodeVisible(n.label)) return; const p=project(nodeW(n));const d=Math.hypot(p.x-mx,p.y-my);if(d<bd){bd=d;best={type:'node',id:n.label};}});
  if(picking){
    if(best){ togglePickNode(best.id); }
    return;
  }
  let bestEl=null, be=8;
  M.elements.forEach(el=>{ if(!elVisible(el)) return; const n1=findNode(el.n1),n2=findNode(el.n2);if(!n1||!n2)return;const a=project(nodeW(n1)),b=project(nodeW(n2));const d=segDist(mx,my,a.x,a.y,b.x,b.y);if(d<be){be=d;bestEl=el;}});
  if(!best && bestEl){
    best={type:'element',id:bestEl.id};
    const lbl=Number(bestEl.label);
    if(additive){ SEL.has(lbl)? SEL.delete(lbl) : SEL.add(lbl); } else SEL=new Set([lbl]);
    syncSelection();
  } else if(!best && !additive && SEL.size){ SEL=new Set(); syncSelection(); }
  PICK=best; scheduleDraw(); renderInspector();
}
/* window selection: elements whose both end nodes lie inside the screen rectangle */
function boxSelect(x0,y0,x1,y1,add){
  if(!add) SEL=new Set();
  const inside=p=>p.x>=x0&&p.x<=x1&&p.y>=y0&&p.y<=y1;
  M.elements.forEach(el=>{ if(!elVisible(el)) return; const n1=findNode(el.n1),n2=findNode(el.n2); if(!n1||!n2) return;
    if(inside(project(nodeW(n1)))&&inside(project(nodeW(n2)))) SEL.add(Number(el.label)); });
  syncSelection(); toast(`${SEL.size} element(s) selected`,'ok');
}
function segDist(px,py,x1,y1,x2,y2){const A=px-x1,B=py-y1,C=x2-x1,D=y2-y1;const dt=A*C+B*D,len=C*C+D*D||1;let t=Math.max(0,Math.min(1,dt/len));const xx=x1+t*C,yy=y1+t*D;return Math.hypot(px-xx,py-yy);}

/* ---- interactive node picking into Supports / Features / Loads node fields ---- */
const PICK_FIELDS='#conList input.mn, #conList input.sn, #featList input.nd, #forceList input.nd';
function setPickTarget(inp){
  if(PICKTARGET===inp) return;
  if(PICKTARGET) PICKTARGET.classList.remove('picking');
  PICKTARGET=inp; inp.classList.add('picking');
  const lbl=inp.closest('.row')? inp.closest('.row').querySelector('label') : null;
  $('#pickBanner').innerHTML=`Click nodes to add/remove them in <b>${esc(lbl?lbl.textContent:'node list')}</b> · <a id="pickDone">done</a> (Esc)`;
  $('#pickBanner').classList.add('show'); $('#pickDone').onclick=clearPickTarget;
  scheduleDraw();
}
function clearPickTarget(){
  if(PICKTARGET) PICKTARGET.classList.remove('picking');
  PICKTARGET=null; $('#pickBanner').classList.remove('show'); scheduleDraw();
}
function togglePickNode(label){
  const inp=PICKTARGET; const cur=parseInts(inp.value);
  const next= cur.includes(+label)? cur.filter(x=>x!==+label) : cur.concat([+label]);
  inp.value=compactInts(next);
  inp.dispatchEvent(new Event('input',{bubbles:true}));
  // the list may have been re-rendered by the input handler: re-acquire the field
  if(!inp.isConnected) clearPickTarget(); else scheduleDraw();
}
function compactInts(a){ // [1,2,3,7] -> "1-3,7"
  const s=[...new Set(a)]; const out=[]; let i=0;
  while(i<s.length){ let j=i; while(j+1<s.length && s[j+1]===s[j]+1 && s[j+1]>s[i]) j++;
    out.push(j-i>=2? `${s[i]}-${s[j]}` : (j>i? `${s[i]},${s[j]}` : `${s[i]}`)); i=j+1; }
  return out.join(',');
}
document.addEventListener('focusin',e=>{
  const t=e.target;
  if(t.matches && t.matches(PICK_FIELDS)) setPickTarget(t);
  else if(t.matches && t.matches('input,select,textarea') && PICKTARGET && t!==PICKTARGET) clearPickTarget();
});
window.addEventListener('keydown',e=>{ if(e.key==='Escape') clearPickTarget(); });

/* view buttons */
$('#vFit').onclick=()=>fitView();
$('#vTop').onclick=()=>{CAM.az=-Math.PI/2;CAM.el=Math.PI/2-1e-4;fitView();};
$('#vFront').onclick=()=>{CAM.az=-Math.PI/2;CAM.el=0;fitView();};
$('#vSide').onclick=()=>{CAM.az=0;CAM.el=0;fitView();};
$('#vIso').onclick=()=>{CAM.az=-0.9;CAM.el=0.5;fitView();};
function toggleBtn(id,key,after){ $('#'+id).onclick=()=>{ VIEW[key]=!VIEW[key]; $('#'+id).classList.toggle('on',VIEW[key]); if(after) after(); scheduleDraw(); }; $('#'+id).classList.toggle('on',VIEW[key]); }
toggleBtn('tLabels','labels');toggleBtn('tSup','supports');toggleBtn('tLoads','loads');
toggleBtn('tDef','deformed');toggleBtn('tEAxes','eaxes',()=>updateLegend());toggleBtn('tCsys','csys');toggleBtn('tDiag','diagrams',()=>updateLegend());
toggleBtn('tResult','result',()=>{ renderResults(); updateLegend(); });
$('#vScaleBtn').onclick=()=>$('#scaleBox').classList.toggle('show');
$('#scaleRange').oninput=()=>{ const v=parseFloat($('#scaleRange').value)||0; $('#scaleVal').textContent=(Math.pow(10,v)).toFixed(v<0?2:v>2?0:2)+'×'; scheduleDraw(); };
$('#autoScale').onchange=scheduleDraw;
function updateBaseScaleRow(){ $('#baseScaleRow').style.display= (RESULT && RESULT.base && isModal() && baseView()==='both')? 'flex' : 'none'; }
$('#baseView').onchange=()=>{ updateBaseScaleRow(); updateLegend(); scheduleDraw(); };
$('#baseScale').oninput=scheduleDraw;
$('#btnAnim').onclick=toggleAnim;
function toggleAnim(){ ANIM.on=!ANIM.on; $('#btnAnim').textContent=ANIM.on?'⏸ Stop':'▶ Animate'; if(ANIM.on){ const step=()=>{ANIM.phase+=0.08; draw(); if(ANIM.on)ANIM.raf=requestAnimationFrame(step);}; step(); } else { cancelAnimationFrame(ANIM.raf); scheduleDraw(); } }


/* ---- selection panel (Model tab) ---- */
function syncSelection(){   // view/selection changed -> list field, badge, table, drawing
  const list=[...SEL].sort((a,b)=>a-b);
  const inp=$('#selList'); if(document.activeElement!==inp) inp.value=compactInts(list);
  $('#nSel').textContent=SEL.size;
  updateVisInfo();
  if($('#selTable').checked) renderEls();
  scheduleDraw();
}
function updateVisInfo(){
  const nvis=ISO? M.elements.filter(elVisible).length : M.elements.length;
  $('#visInfo').textContent= ISO? `Showing ${nvis} of ${M.elements.length} elements.` : '';
  $('#tIsolate').classList.toggle('on',!!ISO);
  updateLegend();
}
function setVisible(set){ ISO=set; VISNODES=null; PICK=null; updateVisInfo(); renderInspector(); scheduleDraw(); }
function renderSelPanel(){
  const sel=$('#selSec'), cur=sel.value;
  const counts={}; M.elements.forEach(e=>{ counts[e.section]=(counts[e.section]||0)+1; });
  sel.innerHTML=M.sections.map(s=>`<option value="${esc(s.name)}">${esc(s.name)} (${counts[s.name]||0})</option>`).join('');
  if(M.sections.some(s=>s.name===cur)) sel.value=cur;
  $('#nSel').textContent=SEL.size; updateVisInfo();
}
function selectBySection(add){
  const name=$('#selSec').value; if(!name) return;
  if(!add) SEL=new Set();
  let n=0; M.elements.forEach(e=>{ if(e.section===name){ SEL.add(Number(e.label)); n++; } });
  syncSelection(); toast(`${n} element(s) with section “${name}” ${add?'added':'selected'}`,'ok');
}
$('#selList').addEventListener('change',e=>{
  const valid=new Set(M.elements.map(el=>Number(el.label)));
  SEL=new Set(parseInts(e.target.value).filter(l=>valid.has(l)));
  e.target.value=compactInts([...SEL].sort((a,b)=>a-b)); syncSelection();
});
$('#selSecSet').onclick=()=>selectBySection(false);
$('#selSecAdd').onclick=()=>selectBySection(true);
$('#selIsolate').onclick=()=>{ if(!SEL.size){ toast('Select elements first (list, section or view)','err'); return; } setVisible(new Set(SEL)); fitView(); };
$('#selHide').onclick=()=>{ if(!SEL.size) return; const vis=new Set(M.elements.filter(elVisible).map(e=>Number(e.label)).filter(l=>!SEL.has(l))); SEL=new Set(); setVisible(vis); syncSelection(); };
$('#selShowAll').onclick=()=>{ setVisible(null); };
$('#selInvert').onclick=()=>{ SEL=new Set(M.elements.filter(elVisible).map(e=>Number(e.label)).filter(l=>!SEL.has(l))); syncSelection(); };
$('#selClear').onclick=()=>{ SEL=new Set(); syncSelection(); };
$('#selTable').onchange=()=>{ SHOWALL.els=false; renderEls(); };
$('#tIsolate').onclick=()=>{ if(ISO) setVisible(null); else if(SEL.size){ setVisible(new Set(SEL)); fitView(); } else toast('Select elements first (click, ctrl-drag, list or section)','err'); };


/* ===================== REFINE MESH (beef ElDef.refine_mesh) ===================== */
let REFINE_UNDO=null;
function openRefine(){
  const nSel=SEL.size;
  const html=`<div class="modal show" id="refineBox"><div class="box" style="max-width:480px">
    <button class="closex" data-closerf>×</button><h2>Refine mesh</h2>
    <p>Subdivides elements with beef's <code>ElDef.refine_mesh</code>: existing node labels are kept, new nodes are numbered from the largest node label upwards, and all elements are renumbered. Sections, element types and parts are inherited; element-load lists and the selection are updated.</p>
    <div class="row"><label>Divisions per element</label><input id="rf_n" type="number" min="2" value="2" class="grow"></div>
    <div class="row"><label>Elements</label><select id="rf_scope" class="grow">
      <option value="all">all elements (${M.elements.length})</option>
      <option value="sel" ${nSel?'selected':''} ${nSel?'':'disabled'}>selected elements (${nSel})</option>
      <option value="list">list…</option></select></div>
    <div class="row" id="rf_listrow" style="display:none"><label>Element list</label><input id="rf_list" type="text" class="grow" placeholder="e.g. 1-20,55"></div>
    <div class="tip" id="rf_info"></div>
    <div class="btnrow"><button class="btn pri" id="rf_go">Refine</button></div>
  </div></div>`;
  const wrap=document.createElement('div'); wrap.innerHTML=html; document.body.appendChild(wrap.firstChild);
  const close=()=>{ const b=$('#refineBox'); if(b) b.remove(); };
  $('[data-closerf]').onclick=close;
  const labels=()=>{ const sc=$('#rf_scope').value;
    if(sc==='sel') return [...SEL];
    if(sc==='list'){ const valid=new Set(M.elements.map(e=>Number(e.label))); return parseInts($('#rf_list').value).filter(l=>valid.has(l)); }
    return null; };
  const info=()=>{ const n=Math.max(2,parseInt($('#rf_n').value)||2), l=labels(), k=l? l.length : M.elements.length;
    $('#rf_listrow').style.display= $('#rf_scope').value==='list'? 'flex' : 'none';
    $('#rf_info').textContent=`${k} element(s) → ${k*n}; elements in model: ${M.elements.length} → ${M.elements.length-k+k*n}; new nodes: ${k*(n-1)}.`; };
  $('#rf_n').oninput=info; $('#rf_scope').onchange=info; $('#rf_list').oninput=info; info();
  $('#rf_go').onclick=async()=>{
    const n=Math.max(2,parseInt($('#rf_n').value)||2), l=labels();
    if(l && !l.length){ toast('No elements to refine','err'); return; }
    $('#rf_go').disabled=true;
    try{ const r=await runJob('refine',{model:modelPayload(), n, element_labels:l}); applyRefine(r); close(); }
    catch(e){ toast(e.message,'err'); $('#rf_go').disabled=false; }
  };
}
function applyRefine(r){
  REFINE_UNDO={nodes:clone(M.nodes), elements:clone(M.elements), element_loads:clone(M.element_loads||[]), sel:[...SEL], iso:ISO?[...ISO]:null};
  const map=r.map, expand=labels=>labels.flatMap(l=>map[String(l)]||[]);
  const before=M.elements.length;
  M.nodes=r.nodes.map(n=>Object.assign({id:uid()},n));
  M.elements=r.elements.map(e=>Object.assign({id:uid()},e));
  (M.element_loads||[]).forEach(e=>{ if(e.target==='list') e.elements=compactInts(expand(parseInts(e.elements)).sort((a,b)=>a-b)); });
  SEL=new Set(expand([...SEL])); if(ISO) ISO=new Set(expand([...ISO]));
  VISNODES=null; PICK=null;
  if(RESULT){ RESULT=null; renderResultsIdle(); $('#tResult').disabled=true; }
  $('#undoRefine').style.display='';
  redrawAll(); syncSelection(); updateLegend();
  toast(`Refined: ${before} → ${M.elements.length} elements, ${M.nodes.length} nodes`,'ok');
}
$('#refineEls').onclick=()=>{ if(!M.elements.length){ toast('No elements','err'); return; } openRefine(); };
$('#undoRefine').onclick=()=>{
  if(!REFINE_UNDO) return;
  M.nodes=REFINE_UNDO.nodes; M.elements=REFINE_UNDO.elements; M.element_loads=REFINE_UNDO.element_loads;
  SEL=new Set(REFINE_UNDO.sel); ISO=REFINE_UNDO.iso? new Set(REFINE_UNDO.iso) : null; VISNODES=null; PICK=null;
  REFINE_UNDO=null; $('#undoRefine').style.display='none';
  if(RESULT){ RESULT=null; renderResultsIdle(); $('#tResult').disabled=true; }
  redrawAll(); syncSelection(); updateLegend(); toast('Refinement undone','ok');
};

/* ===================== SERVER API ===================== */
const QS=new URLSearchParams(location.search);
if(QS.get('token')) sessionStorage.setItem('beef_token',QS.get('token'));
const TOKEN=QS.get('token')||sessionStorage.getItem('beef_token')||'';
let INFO={large_dofs:1500, sparse_kinds:['eig','lin_static']};
class ApiError extends Error{ constructor(msg,detail){ super(msg); this.detail=detail||{}; } }
async function api(name,params={}){
  let r;
  try{ r=await fetch('/api/'+name,{method:'POST',headers:{'Content-Type':'application/json','X-Token':TOKEN},body:JSON.stringify(params)}); }
  catch(e){ throw new ApiError('Lost connection to the BEEF GUI server. Is it still running in the terminal?'); }
  let j; try{ j=await r.json(); }catch(e){ throw new ApiError('Server returned '+r.status); }
  if(!j.ok) throw new ApiError(j.error||'Unknown error', j);
  return j.data;
}
/* the model as sent to the server (without GUI-only ids) */
function modelPayload(){ return JSON.parse(JSON.stringify(M,(k,v)=> k==='id'? undefined : v)); }

/* ===================== JOBS (log drawer) ===================== */
const Jobs={ sel:null, timer:null, waiters:{}, list:[] };
async function pollJobs(){
  try{
    const d=await api('jobs'); Jobs.list=d.jobs; renderJobList();
    if(Jobs.sel!=null) await showJob(Jobs.sel);
    for(const id of Object.keys(Jobs.waiters)){
      const j=d.jobs.find(x=>x.id==id);
      if(j && !['queued','running'].includes(j.status)){
        const w=Jobs.waiters[id]; delete Jobs.waiters[id];
        try{ const full=await api('job',{id:+id,result:true});
          if(full.status==='done') w.resolve(full.result); else w.reject(new ApiError(full.error||full.status,{traceback:full.traceback,log:full.log})); }
        catch(e){ w.reject(e); }
      }
    }
    const busy=d.busy||Object.keys(Jobs.waiters).length;
    clearTimeout(Jobs.timer); Jobs.timer=setTimeout(pollJobs, busy? 400 : 3000);
  }catch(e){ setStatus(e.message,'err'); clearTimeout(Jobs.timer); Jobs.timer=setTimeout(pollJobs,5000); }
}
function waitJob(id){ return new Promise((resolve,reject)=>{ Jobs.waiters[id]={resolve,reject}; clearTimeout(Jobs.timer); Jobs.timer=setTimeout(pollJobs,200); }); }
async function runJob(name,params){
  const d=await api(name,params);
  if(d && typeof d==='object' && 'job' in d && 'done' in d){
    if(d.job!=null){ Jobs.sel=d.job; }
    pollJobs();
    if(d.done) return d.result;
    return waitJob(d.job);
  }
  return d;
}
function renderJobList(){
  const box=$('#jobList'); if(!box) return;
  box.innerHTML=Jobs.list.map(j=>`<div class="job ${j.id===Jobs.sel?'sel':''} st-${j.status}" data-id="${j.id}"><span class="jdot"></span><b>#${j.id}</b> ${esc(j.label)}<span class="jt">${j.elapsed!=null?j.elapsed.toFixed(1)+' s':''}</span></div>`).join('')||'<div class="tip">No jobs yet.</div>';
  $$('.job',box).forEach(x=>x.onclick=()=>{ Jobs.sel=+x.dataset.id; renderJobList(); showJob(Jobs.sel); });
  const cur=Jobs.list.find(j=>j.status==='running')||Jobs.list[0];
  if(cur){ $('#drawerStatus').textContent=`#${cur.id} ${cur.label}: ${cur.status}${cur.last_line? ' — '+cur.last_line:''}`;
    $('#drawerDot').className='dot '+({running:'busy',queued:'busy',done:'ok',error:'err',cancelled:'err'}[cur.status]||''); }
}
async function showJob(id){
  try{ const j=await api('job',{id}); const pre=$('#jobLog'); const atEnd=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-8;
    $('#jobTitle').textContent=`#${j.id} ${j.label} — ${j.status}${j.elapsed!=null?' ('+j.elapsed.toFixed(2)+' s)':''}`;
    pre.textContent=(j.log||'')+(j.error&&!j.log.includes(j.error)?'\n'+j.error:'');
    if(atEnd) pre.scrollTop=pre.scrollHeight;
    $('#jobCancel').disabled=!['running','queued'].includes(j.status);
  }catch(e){ /* job cleared */ }
}
$('#drawerHead').onclick=()=>$('#drawer').classList.toggle('open');
$('#jobCancel').onclick=async()=>{ if(Jobs.sel!=null){ try{ await api('job/cancel',{id:Jobs.sel}); }catch(e){ toast(e.message,'err'); } pollJobs(); } };
$('#jobClear').onclick=async()=>{ await api('jobs/clear'); Jobs.sel=null; $('#jobLog').textContent=''; pollJobs(); };
$('#consoleIn').addEventListener('keydown',async e=>{
  if(e.key!=='Enter'||e.shiftKey) return; e.preventDefault();
  const code=e.target.value; if(!code.trim()) return; e.target.value='';
  try{ await runJob('console',{code}); }catch(err){ /* error is in the job log */ }
});

/* ===================== RUN ===================== */
function setStatus(txt,cls){ $('#statTxt').textContent=txt; $('#statDot').className='dot '+(cls||''); }
let RESULT=null;
async function runAnalysis(){
  if(!M.nodes.length||!M.elements.length){ toast('Add nodes and elements first','err'); return; }
  const btn=$('#btnRun'); btn.disabled=true; const kind=M.analysis.kind;
  setStatus('Running '+kind+' …','busy'); $('#drawer').classList.add('open');
  try{
    const data=await runJob('run',{model:modelPayload(), solver:$('#solverSel').value});
    RESULT=prepResult(data); VIEW.result=true; $('#tResult').classList.add('on');
    setStatus(`Done — ${kind} (${data.solver} solver)`,'ok');
    afterRun();
  }catch(e){
    setStatus('Analysis failed','err'); showError({error:e.message, trace:(e.detail&&(e.detail.traceback||e.detail.log))||''});
  }
  btn.disabled=false;
}
function prepResult(d){ d.selMode=0; d.timeIndex=(d.u? d.u[0].length-1 : null); return d; }
function afterRun(){
  VIEW.deformed=true; $('#tDef').classList.add('on'); $('#scaleBox').classList.add('show');
  if(RESULT.elements && !RESULT.elements.keys.includes(DIAG)) DIAG = RESULT.domain==='2d'?'M':'Mz';
  $('#scaleRange').value=0; $('#scaleVal').textContent='1×';
  $('#baseBox').style.display= RESULT.base? 'block' : 'none';
  if(RESULT.base && $('#baseView').options && $('#baseView').options.length>=3){ const o=$('#baseView').options, modal=isModal();
    o[0].text= modal? 'base + mode' : 'total (base + increment)'; o[1].text='base state only'; o[2].text= modal? 'mode only' : 'increment only'; }
  updateBaseScaleRow();
  $('#tResult').disabled=false;
  renderResults(); updateLegend(); scheduleDraw();
}
/* switching analysis type hides the previous result (toggle "Result" to see it again until the next run) */
function hideResult(){ if(!RESULT) return; VIEW.result=false; $('#tResult').classList.remove('on'); if(ANIM.on) toggleAnim(); renderResults(); updateLegend(); scheduleDraw(); }

/* ===================== RESULTS PANEL ===================== */
function renderResultsIdle(){ $('#resultsBox').innerHTML=`<div class="hint">Results appear here after you run an analysis. Analyses run in the local Python session with the installed <code>beef</code>; progress and output are shown in the log drawer at the bottom.</div>`; }
function showError(d){
  $('#resultsBox').innerHTML=`<div class="warn">Analysis failed.</div>
    <div class="kv"><span>Message</span></div><pre class="code" style="max-height:140px;white-space:pre-wrap">${esc(d.error||'')}</pre>
    ${d.trace?`<details open><summary style="color:var(--muted);cursor:pointer;font-size:11px">Python traceback / log</summary><pre class="code" style="max-height:260px">${esc(d.trace)}</pre></details>`:''}`;
}
function fmt(v){ if(v==null||!isFinite(v))return '—'; const a=Math.abs(v); if(a!==0&&(a<1e-3||a>=1e5)) return v.toExponential(3).replace('e+','e'); return (Math.round(v*1e4)/1e4).toString(); }
function renderResults(){
  if(!RESULT){ renderResultsIdle(); return; }
  const k=RESULT.kind; let html='';
  if(!VIEW.result) html+=`<div class="warn">Showing the previous <b>${esc(k)}</b> result is switched off (analysis type changed). Toggle <b>Result</b> in the view bar to show it, or run the new analysis.</div>`;
  html+=`<div class="kv"><span>Solver</span><b>${RESULT.solver==='sparse'?'sparse':'beef'} · ${RESULT.ndofs} DOFs · beef ${esc(RESULT.beef_version||'')}</b></div>`;
  if(RESULT.note) html+=`<div class="warn">${esc(RESULT.note)}</div>`;
  if(RESULT.base){ const B=RESULT.base; let nmx=0; B.elements.values.N.forEach(v=>{ nmx=Math.max(nmx,Math.abs(v[0]),Math.abs(v[1])); });
    html+=`<div class="kv"><span>Base state</span><b>${B.mode==='static'?'nonlinear static':'linear static'} · max |u| ${fmt(B.max_u)} · max |N| ${fmt(nmx)}</b></div>`; }
  $('#resultsBox').innerHTML=html+'<div id="resBody"></div>';
  if(k==='eig') renderModal(RESULT,false);
  else if(k==='buckling') renderModal(RESULT,true);
  else if(k==='lin_dynamic'||k==='dynamic') renderDynamic(RESULT);
  else renderStatic(RESULT);
}
function diagLabel(k){ return ({N:'N (axial)',Q:'V (shear)',M:'M (moment)',Qy:'Vy',Qz:'Vz',Mx:'T (Mx)',My:'My',Mz:'Mz'})[k]||k; }
function diagramBlock(R){
  if(!R.elements) return '';
  const E=R.elements;
  let html=`<h3 class="sect" style="margin-top:12px">Section forces</h3><div class="subtabs" id="diagTabs">`+
    E.keys.map(kk=>`<button class="subtab ${kk===DIAG?'active':''}" data-dg="${kk}">${diagLabel(kk)}</button>`).join('')+`</div>`;
  html+=`<table class="grid"><thead><tr><th>Comp.</th><th>min</th><th>max</th><th>at elem.</th></tr></thead><tbody>`;
  E.keys.forEach(kk=>{ let mn=Infinity,mx=-Infinity,arg=null,am=-1;
    E.values[kk].forEach((v,i)=>v.forEach(x=>{ mn=Math.min(mn,x); mx=Math.max(mx,x); if(Math.abs(x)>am){am=Math.abs(x);arg=E.labels[i];} }));
    html+=`<tr class="${kk===DIAG?'selrow':''}"><td>${diagLabel(kk)}</td><td>${fmt(mn)}</td><td>${fmt(mx)}</td><td>${arg}</td></tr>`; });
  return html+`</tbody></table><div class="tip">Values in model units. Diagrams are drawn in the element's local plane (Vy, Mz along local y; Vz, My along local z).</div>`;
}
function wireDiagTabs(R){
  $$('#diagTabs .subtab').forEach(b=>b.onclick=()=>{ DIAG=b.dataset.dg; VIEW.diagrams=true; $('#tDiag').classList.add('on'); renderResults(); updateLegend(); scheduleDraw(); });
}
function renderStatic(R){
  const col=R.u.map(row=>row[R.timeIndex!=null?R.timeIndex:row.length-1]);
  let maxd=0,maxlbl='',maxdir='';
  R.nodeLabels.forEach((lbl,i)=>{ const gd=R.node_gdofs[i]; const td=R.domain==='2d'?2:3; for(let j=0;j<td;j++){const v=Math.abs(col[gd[j]]); if(v>maxd){maxd=v;maxlbl=lbl;maxdir=DOFNAMES[R.domain][j];}} });
  let html=`<div class="kv"><span>Analysis</span><b>${R.kind==='static'?'Nonlinear static':'Linear static'}</b></div>`;
  html+=`<div class="kv"><span>Max |translation|</span><b>${fmt(maxd)}</b></div><div class="kv"><span>at node / dir</span><b>${maxlbl} · ${maxdir}</b></div>`;
  html+=diagramBlock(R);
  if(R.reactions&&R.reactions.length){
    html+=`<h3 class="sect" style="margin-top:12px">Reactions</h3><table class="grid"><thead><tr><th>Node</th><th>DOF</th><th>Value</th></tr></thead><tbody>`;
    R.reactions.forEach(r=>{ if(r.value!=null&&Math.abs(r.value)>1e-9) html+=`<tr><td>${r.node}</td><td>${DOFNAMES[R.domain][r.dof]||r.dof}</td><td>${fmt(r.value)}</td></tr>`; });
    html+=`</tbody></table>`;
  }
  html+=`<div id="inspector"></div>`;
  $('#resBody').innerHTML=html; wireDiagTabs(R); renderInspector();
}
function renderModal(R,isBuck){
  const arr=isBuck?R.factors:R.frequencies;
  let html=`<div class="kv"><span>${isBuck?'Buckling analysis':'Modal analysis'}</span><b>${arr.length} modes</b></div>`;
  html+=`<div class="hint">Click a mode to view its shape. Use Scale… to animate.</div><div class="modelist">`;
  arr.forEach((v,i)=>{
    const label=isBuck? `λ = ${fmt(v)}` : `${fmt(v)} Hz`;
    const sub=isBuck? `mode ${i+1}` : (R.damping&&Math.abs(R.damping[i])>1e-6?`ζ=${(R.damping[i]*100).toFixed(2)}%`:(v>0?`T=${fmt(1/v)} s`:`mode ${i+1}`));
    html+=`<div class="mode ${i===R.selMode?'sel':''}" data-mi="${i}"><span><b>${i+1}.</b> ${label}</span><span style="color:var(--muted)">${sub}</span></div>`;
  });
  $('#resBody').innerHTML=html+'</div>';
  $$('#resBody .mode').forEach(m=>m.onclick=()=>{ R.selMode=+m.dataset.mi; VIEW.deformed=true; renderModal(R,isBuck); scheduleDraw(); });
}
let ELTIMER=null;
function renderDynamic(R){
  let html=`<div class="kv"><span>${R.kind==='dynamic'?'Nonlinear dynamic':'Linear dynamic'}</span><b>${R.t.length} steps</b></div>`;
  html+=`<div class="kv"><span>Duration</span><b>${fmt(R.t[R.t.length-1])} s</b></div>`;
  html+=`<h3 class="sect" style="margin-top:10px">Response history</h3>`;
  html+=`<div class="row"><label>Node</label><input id="thNode" class="grow" type="number" value="${R._thNode||R.nodeLabels[Math.floor(R.nodeLabels.length/2)]}"></div>`;
  html+=`<div class="row"><label>DOF</label><select id="thDof" class="grow">${DOFNAMES[R.domain].map((n,i)=>`<option value="${i}" ${i==(R._thDof!=null?R._thDof:(R.domain==='2d'?1:2))?'selected':''}>${i} — ${n}</option>`).join('')}</select></div>`;
  html+=`<canvas id="thplot" height="170"></canvas>`;
  html+=`<h3 class="sect" style="margin-top:12px">Deformed shape @ t</h3>`;
  html+=`<div class="row"><input type="range" id="timeSlider" min="0" max="${R.t.length-1}" value="${R.timeIndex}" style="width:100%"></div>`;
  html+=`<div class="kv"><span>t</span><b id="tNow">${fmt(R.t[R.timeIndex])} s</b></div>`;
  html+=diagramBlock(R)+`<div id="inspector"></div>`;
  $('#resBody').innerHTML=html; wireDiagTabs(R);
  const upd=()=>{ R._thNode=+$('#thNode').value; R._thDof=+$('#thDof').value; drawTH(R,R._thNode,R._thDof); };
  $('#thNode').oninput=upd; $('#thDof').onchange=upd; upd();
  $('#timeSlider').oninput=e=>{ R.timeIndex=+e.target.value; $('#tNow').textContent=fmt(R.t[R.timeIndex])+' s'; scheduleDraw();
    clearTimeout(ELTIMER); ELTIMER=setTimeout(async()=>{ try{ R.elements=await api('element_results',{time_index:R.timeIndex}); scheduleDraw(); }catch(err){} },250); };
}
function drawTH(R,nodeLbl,dof){
  const cv2=$('#thplot'); if(!cv2)return; const w=cv2.clientWidth||280,h=170; cv2.width=w*DPR;cv2.height=h*DPR; const c=cv2.getContext('2d');c.setTransform(DPR,0,0,DPR,0,0);
  const i=R.nodeLabels.indexOf(nodeLbl); c.clearRect(0,0,w,h); if(i<0){ c.fillStyle='#8aa0b6'; c.fillText('no such node',10,20); return; }
  const g=R.node_gdofs[i][dof]; const ys=R.u[g], xs=R.t;
  let ymin=Infinity,ymax=-Infinity; ys.forEach(v=>{ymin=Math.min(ymin,v);ymax=Math.max(ymax,v);}); if(ymin===ymax){ymin-=1;ymax+=1;}
  const x0=30,x1=w-8,y0=h-20,y1=8;
  const X=t=>x0+(t-xs[0])/((xs[xs.length-1]-xs[0])||1)*(x1-x0), Y=v=>y0+(v-ymin)/((ymax-ymin)||1)*(y1-y0);
  c.strokeStyle='#263040';c.beginPath();c.moveTo(x0,Y(0));c.lineTo(x1,Y(0));c.stroke();
  c.strokeStyle=COL.deformed;c.lineWidth=1.6;c.beginPath();xs.forEach((t,k)=>{const px=X(t),py=Y(ys[k]);k?c.lineTo(px,py):c.moveTo(px,py);});c.stroke();
  c.fillStyle='#8aa0b6';c.font='10px ui-monospace,Consolas,monospace';c.fillText(fmt(ymax),2,12);c.fillText(fmt(ymin),2,y0);c.fillText('t [s]',w/2-8,h-4);
  c.fillText(`${DOFNAMES[R.domain][dof]} node ${nodeLbl}`,x0,12);
}
function renderInspector(){
  const box=$('#inspector'); if(!box) return;
  if(!PICK){ box.innerHTML=`<div class="tip" style="margin-top:10px">Click a node or element in the view to inspect it.</div>`; return; }
  let html=`<h3 class="sect" style="margin-top:12px">Inspector</h3>`;
  if(PICK.type==='node'){
    const n=findNode(PICK.id); if(!n){box.innerHTML='';return;}
    html+=`<div class="kv"><span>Node</span><b>${n.label}</b></div><div class="kv"><span>Coord</span><b>${fmt(+n.x)}, ${fmt(+n.y)}${M.domain==='2d'?'':', '+fmt(+n.z)}</b></div>`;
    if(RESULT&&RESULT.u){ const i=RESULT.nodeLabels.indexOf(n.label); if(i>=0){ const ti=RESULT.timeIndex!=null?RESULT.timeIndex:RESULT.u[0].length-1; const gd=RESULT.node_gdofs[i];
      DOFNAMES[RESULT.domain].forEach((nm,j)=>{ html+=`<div class="kv"><span>${nm}</span><b>${fmt(RESULT.u[gd[j]][ti])}</b></div>`; }); } }
  } else {
    const el=M.elements.find(e=>e.id===PICK.id); if(!el){box.innerHTML='';return;}
    html+=`<div class="kv"><span>Element</span><b>[${el.label}] ${el.n1}→${el.n2}</b></div><div class="kv"><span>Section</span><b>${esc(el.section)}</b></div><div class="kv"><span>Type</span><b>${el.etype}</b></div>`;
    const E=RESULT&&RESULT.elements; if(E){ const ei=E.labels.indexOf(Number(el.label)); if(ei>=0) E.keys.forEach(kk=>{ const v=E.values[kk][ei]; html+=`<div class="kv"><span>${diagLabel(kk)} (end 1 / end 2)</span><b>${fmt(v[0])} / ${fmt(v[1])}</b></div>`; }); }
  }
  box.innerHTML=html;
}
function updateLegend(){
  const items=[`<div><i style="background:${COL.member}"></i> element</div>`];
  if(resultShown()&&(VIEW.deformed||isModal())) items.push(`<div><i style="background:${COL.deformed}"></i> ${baseView()==='base'?'static base state':'deformed'}</div>`);
  if(resultShown()&&isModal()&&baseView()==='both') items.push(`<div><span style="color:${COL.member}">┄</span> base state</div>`);
  items.push(`<div><span style="color:${COL.support}">▲</span> support</div>`);
  if(M.forces.length) items.push(`<div><span style="color:${COL.load}">→</span> load</div>`);
  if(VIEW.diagrams&&resultShown()&&RESULT.elements) items.push(`<div><i style="background:${COL.diag}"></i> ${diagLabel(DIAG)}</div>`);
  if(RESULT&&!VIEW.result) items.push(`<div style="color:var(--muted)">result hidden</div>`);
  if(SEL.size) items.push(`<div><i style="background:${COL.sel}"></i> selected (${SEL.size})</div>`);
  if(VIEW.eaxes) items.push(`<div>local <span style="color:${AXCOL[0]}">x</span> <span style="color:${AXCOL[1]}">y</span>${M.domain==='2d'?'':` <span style="color:${AXCOL[2]}">z</span>`}</div>`);
  if(ISO) items.push(`<div style="color:var(--muted)">showing ${M.elements.filter(elVisible).length}/${M.elements.length} elements</div>`);
  $('#legend').innerHTML=items.join('');
}

/* ===================== PYTHON SCRIPT ===================== */
async function showCode(){
  try{
    const d=await api('script',{model:modelPayload()});
    openCodeModal(d.script);
  }catch(e){ toast(e.message,'err'); }
}
function openCodeModal(code){
  $('#codePre').innerHTML=highlight(code.length>300000? code.slice(0,300000)+'\n# … (truncated in view; use Download)' : code);
  $('#codeModal').classList.add('show');
  $('#cpCode').onclick=()=>{ navigator.clipboard.writeText(code).then(()=>toast('Copied','ok'),()=>toast('Copy failed','err')); };
  $('#dlCode').onclick=()=>saveFile('beef_model.py', code, 'text/x-python');
}
function highlight(code){
  let s=esc(code);
  s=s.replace(/(#[^\n]*)/g,'<span class="tok-com">$1</span>');
  s=s.replace(/\b(import|from|as|def|return|for|in|if|else|elif|try|except|True|False|None|with|class|and|or|not)\b/g,'<span class="tok-kw">$1</span>');
  s=s.replace(/\b(fe|np|analysis|part|assembly|eldef)\.([a-zA-Z_]+)/g,'$1.<span class="tok-fn">$2</span>');
  return s;
}
$('#btnCode').onclick=showCode;
$('#btnRun').onclick=runAnalysis;

/* ===================== EXAMPLES ===================== */
async function renderExamples(){
  try{
    const d=await api('examples'), list=d.examples;
    $('#exDirs').textContent=d.dirs.join(' · ');
    $('#exList').innerHTML=list.map(e=>`<div class="ex" data-ex="${e.index}"><b>${esc(e.name)}</b><span>${esc(e.desc)}</span><span class="tip">${esc(e.file)}${e.nodes!=null?` · ${e.nodes} nodes, ${e.elements} elements`:''}</span></div>`).join('')||'<div class="tip">No .beef files found.</div>';
    $$('#exList .ex').forEach(x=>x.onclick=async()=>{ try{ const m=await api('example',{index:+x.dataset.ex}); loadModel(m); closeModals(); fitView(); toast('Loaded: '+(m.name||'example'),'ok'); }catch(e){ toast(e.message,'err'); } });
  }catch(e){ $('#exList').innerHTML=`<div class="warn">${esc(e.message)}</div>`; }
}

/* ===================== MODALS ===================== */
function closeModals(){ $$('.modal').forEach(m=>m.classList.remove('show')); }
$('#btnHelp').onclick=()=>$('#helpModal').classList.add('show');
$('#btnExamples').onclick=()=>{ renderExamples(); $('#exModal').classList.add('show'); };
$$('[data-close]').forEach(b=>b.onclick=closeModals);
$$('.modal').forEach(m=>m.addEventListener('mousedown',e=>{ if(e.target===m) closeModals(); }));
window.addEventListener('keydown',e=>{ if(e.key==='Escape') closeModals(); });

/* ===================== INIT ===================== */
async function init(){
  M=blankModel('3d');
  $('#domain').value=M.domain; $('#anaKind').value=M.analysis.kind;
  redrawAll(); renderResultsIdle(); updateLegend(); resizeCanvas();
  $('#tResult').disabled=true;
  try{
    INFO=await api('info');
    $('#beefVer').textContent=`beef ${INFO.beef_version}`; $('#beefVer').title=`beef from ${INFO.beef_path}\nPython ${INFO.python} · NumPy ${INFO.numpy} · SciPy ${INFO.scipy}`;
    $('#largeDofs').textContent=INFO.large_dofs;
    setStatus('Connected — beef '+INFO.beef_version,'ok');
    const ex=(await api('examples')).examples; const first=ex.find(e=>/portal/i.test(e.file))||ex[0];
    if(first){ loadModel(await api('example',{index:first.index})); requestAnimationFrame(()=>fitView()); }
  }catch(e){ setStatus(e.message,'err'); }
  pollJobs();
}
init();
/* keep the canvas backing store in sync with layout changes (drawer, window, panels) */
if(window.ResizeObserver) new ResizeObserver(()=>resizeCanvas()).observe(cv);

/* resizable left panel (width remembered in localStorage) */
(function(){
  const root=document.documentElement, sp=$('#lsplit'), KEY='beefgui.leftw', DEF=460;
  const setW=w=>{ w=Math.max(320,Math.min(Math.round(w), Math.max(360,window.innerWidth-420))); root.style.setProperty('--left-w',w+'px'); return w; };
  try{ const w=parseFloat(localStorage.getItem(KEY)); if(w) setW(w); }catch(e){}
  let d=null;
  sp.addEventListener('mousedown',e=>{ d={x:e.clientX,w:$('aside.left').getBoundingClientRect().width}; sp.classList.add('drag'); e.preventDefault(); });
  window.addEventListener('mousemove',e=>{ if(!d) return; setW(d.w+e.clientX-d.x); resizeCanvas(); });
  window.addEventListener('mouseup',()=>{ if(!d) return; d=null; sp.classList.remove('drag');
    try{ localStorage.setItem(KEY, parseFloat(getComputedStyle(root).getPropertyValue('--left-w'))); }catch(e){} });
  sp.addEventListener('dblclick',()=>{ setW(DEF); try{ localStorage.removeItem(KEY); }catch(e){} resizeCanvas(); });
})();
