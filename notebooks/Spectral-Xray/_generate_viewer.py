"""
Generate a self-contained interactive HTML CT viewer for the lizard head.
Embeds the volumes as zlib-compressed base64 and decompresses with pako.js.
"""
import numpy as np, io, base64, zlib, json, os
from scipy.ndimage import gaussian_filter, median_filter

NPZ  = r"E:\git\Spectral-Xray\lizard_recon.npz"
OUT  = r"E:\git\Spectral-Xray\lizard_viewer.html"
TISSUE_LO = 0.0035
BONE_LO   = 0.0130
HEAD_R    = 28.0


def ring_correct(raw):
    vol = median_filter(raw.astype(np.float64), size=3)
    vol = gaussian_filter(vol, sigma=0.8)
    nz, ny, nx = vol.shape
    cy, cx = ny/2-.5, nx/2-.5
    Y, X = np.mgrid[:ny, :nx]
    R = np.hypot(X-cx, Y-cy)
    Ri = np.round(R).astype(int)
    nb = Ri.max()+1
    cnt = np.maximum(np.bincount(Ri.ravel(), None, nb), 1)
    anat = np.empty_like(vol)
    for z in range(nz):
        rm = np.bincount(Ri.ravel(), vol[z].ravel(), nb)/cnt
        anat[z] = vol[z] - rm[Ri]
    anat = np.clip(anat, 0, None)
    anat[:, R > HEAD_R] = 0.0
    anat = gaussian_filter(anat, sigma=0.8)
    hi = np.percentile(anat[anat>0], 99.5)
    return np.clip(anat/hi, 0, 1), hi


def norm_u8(arr):
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn: return np.zeros_like(arr, dtype=np.uint8)
    return ((arr-mn)/(mx-mn)*255).clip(0,255).astype(np.uint8)


def pack(arr):
    """uint8 ndarray → zlib-compressed base64 string"""
    compressed = zlib.compress(arr.tobytes(), level=9)
    return base64.b64encode(compressed).decode()


print("Loading data...", flush=True)
d = np.load(NPZ)
vol_br = d["vol_br"].astype(np.float64)
vol_lo = d["vol_lo"].astype(np.float64)
vol_hi = d["vol_hi"].astype(np.float64)
voxel  = float(d["voxel_mm"])

print("Ring-correcting...", flush=True)
anat, anat_hi = ring_correct(vol_br)

with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(vol_lo > 1e-4, vol_hi/vol_lo, 0.0)

seg = np.zeros(anat.shape, dtype=np.uint8)
seg[anat >= TISSUE_LO/anat_hi] = 1
seg[anat >= BONE_LO/anat_hi]   = 2

print("Packing volumes...", flush=True)
packed = {
    "anat":  pack(norm_u8(anat)),
    "broad": pack(norm_u8(vol_br)),
    "lo":    pack(norm_u8(vol_lo)),
    "hi":    pack(norm_u8(vol_hi)),
    "ratio": pack(norm_u8(ratio)),
    "seg":   pack(seg),
}
nz, ny, nx = anat.shape

sizes = {k: len(v) for k, v in packed.items()}
print("Packed sizes (base64 of zlib):", sizes, flush=True)

data_js = json.dumps({
    "shape": [nz, ny, nx],
    "voxel": voxel,
    "tissue_u8": int(round((TISSUE_LO/anat_hi)*255)),
    "bone_u8":   int(round((BONE_LO/anat_hi)*255)),
    "vols": packed,
})

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lizard-head spectral CT viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#ddd;font-family:monospace;font-size:13px;padding:12px}
h2{color:#eee;font-size:16px;font-weight:500;margin-bottom:10px}
.top{display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap;align-items:center}
button{background:#222;color:#bbb;border:1px solid #444;padding:4px 12px;cursor:pointer;border-radius:4px;font-size:12px}
button.active{background:#1a3a5c;color:#7ec8f4;border-color:#7ec8f4}
.panels{display:flex;gap:6px;flex-wrap:wrap}
.panel{flex:1;min-width:200px}
.panel label{display:block;font-size:11px;color:#888;margin-bottom:3px;text-align:center}
canvas{display:block;width:100%;cursor:crosshair;image-rendering:pixelated;border:1px solid #333;border-radius:3px}
.slider-row{display:flex;gap:6px;align-items:center;margin-top:4px}
.slider-row input{flex:1}
.slider-row span{min-width:26px;text-align:right;font-size:11px;color:#888}
#status{font-size:11px;color:#666;margin-top:8px}
#legend{display:flex;gap:14px;margin-top:8px;font-size:11px;align-items:center}
.lbox{width:14px;height:14px;border-radius:2px;display:inline-block;margin-right:4px;vertical-align:middle}
</style>
</head>
<body>
<h2>Lizard-head spectral CT &mdash; interactive MPR viewer</h2>

<div class="top" id="band-btns">
  <span style="color:#666">Band:</span>
  <button class="active" data-band="anat">anatomy</button>
  <button data-band="broad">broad</button>
  <button data-band="lo">low-E</button>
  <button data-band="hi">high-E</button>
  <button data-band="ratio">hi/lo ratio</button>
  <button data-band="seg">segmented</button>
</div>

<div class="panels">
  <div class="panel">
    <label>axial (z)</label>
    <canvas id="cv-ax" width="80" height="80"></canvas>
    <div class="slider-row"><input type="range" id="sl-ax" min="0" max="79" value="40"><span id="lbl-ax">40</span></div>
  </div>
  <div class="panel">
    <label>coronal (y)</label>
    <canvas id="cv-cor" width="80" height="80"></canvas>
    <div class="slider-row"><input type="range" id="sl-cor" min="0" max="79" value="40"><span id="lbl-cor">40</span></div>
  </div>
  <div class="panel">
    <label>sagittal (x)</label>
    <canvas id="cv-sag" width="80" height="80"></canvas>
    <div class="slider-row"><input type="range" id="sl-sag" min="0" max="79" value="40"><span id="lbl-sag">40</span></div>
  </div>
</div>

<div id="legend">
  <span><span class="lbox" style="background:#888"></span>background</span>
  <span id="leg-t" style="display:none"><span class="lbox" style="background:#d98c6a"></span>soft tissue</span>
  <span id="leg-b" style="display:none"><span class="lbox" style="background:#f2ead6"></span>bone/dense</span>
  <span id="leg-scale"></span>
</div>
<div id="status">Loading...</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<script>
const META = DATA_PLACEHOLDER;

const NZ=META.shape[0], NY=META.shape[1], NX=META.shape[2];
const TU=META.tissue_u8, BU=META.bone_u8;
const VOXEL=META.voxel;

let vols = {};
let band = 'anat';
let ax=40, cor=40, sag=40;

function b64ToU8(s){
  const bin=atob(s); const u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i);
  return u;
}

function decompressVol(key){
  const compressed=b64ToU8(META.vols[key]);
  return pako.inflate(compressed);
}

function idx(z,y,x){ return z*NY*NX + y*NX + x; }

function makeLUT(mode){
  const lut=new Uint8ClampedArray(256*4);
  for(let v=0;v<256;v++){
    let r,g,b;
    if(mode==='seg'){
      if(v===0){r=5;g=5;b=8}
      else if(v===1){r=217;g=140;b=106}
      else{r=242;g=234;b=214}
    } else if(mode==='anat'){
      if(v<TU){
        const t=v/TU;
        r=Math.round(5+t*30); g=Math.round(5+t*30); b=Math.round(8+t*35);
      } else if(v<BU){
        const t=(v-TU)/(BU-TU);
        r=Math.round(30+t*(217-30)); g=Math.round(30+t*(140-30)); b=Math.round(40+t*(106-40));
      } else {
        const t=(v-BU)/(255-BU);
        r=Math.round(217+t*(242-217)); g=Math.round(140+t*(234-140)); b=Math.round(106+t*(214-106));
      }
    } else if(mode==='ratio'){
      // plasma-like
      const t=v/255;
      r=Math.round(13+t*(240-13));
      g=Math.round(8+t*t*(149-8));
      b=Math.round(135+t*(10-135));
    } else {
      // bone-style grayscale
      const t=v/255;
      r=Math.round(t*t*235); g=Math.round(t*230); b=Math.round(t*210);
    }
    lut[v*4]=r; lut[v*4+1]=g; lut[v*4+2]=b; lut[v*4+3]=255;
  }
  return lut;
}

function drawSlice(ctx, slice2d, lut, W, H, crossX, crossY){
  const img=ctx.createImageData(W,H);
  const d=img.data;
  for(let r=0;r<H;r++) for(let c=0;c<W;c++){
    const v=slice2d[r*W+c];
    const pi=(r*W+c)*4, li=v*4;
    d[pi]=lut[li]; d[pi+1]=lut[li+1]; d[pi+2]=lut[li+2]; d[pi+3]=255;
  }
  ctx.putImageData(img,0,0);
  // crosshairs
  ctx.save();
  ctx.strokeStyle='rgba(100,200,255,0.6)';
  ctx.lineWidth=0.5;
  if(crossX>=0){ctx.beginPath();ctx.moveTo(crossX+.5,0);ctx.lineTo(crossX+.5,H);ctx.stroke();}
  if(crossY>=0){ctx.beginPath();ctx.moveTo(0,crossY+.5);ctx.lineTo(W,crossY+.5);ctx.stroke();}
  ctx.restore();
}

function render(){
  const vol=vols[band];
  if(!vol) return;
  const lut=makeLUT(band);

  // axial: z=ax, rows=y, cols=x
  {
    const sl=new Uint8Array(NY*NX);
    for(let y=0;y<NY;y++) for(let x=0;x<NX;x++) sl[y*NX+x]=vol[idx(ax,y,x)];
    drawSlice(document.getElementById('cv-ax').getContext('2d'), sl, lut, NX, NY, sag, cor);
  }
  // coronal: y=cor, rows=z, cols=x
  {
    const sl=new Uint8Array(NZ*NX);
    for(let z=0;z<NZ;z++) for(let x=0;x<NX;x++) sl[z*NX+x]=vol[idx(z,cor,x)];
    drawSlice(document.getElementById('cv-cor').getContext('2d'), sl, lut, NX, NZ, sag, ax);
  }
  // sagittal: x=sag, rows=z, cols=y
  {
    const sl=new Uint8Array(NZ*NY);
    for(let z=0;z<NZ;z++) for(let y=0;y<NY;y++) sl[z*NY+y]=vol[idx(z,y,sag)];
    drawSlice(document.getElementById('cv-sag').getContext('2d'), sl, lut, NY, NZ, cor, ax);
  }
}

function updateInfo(){
  document.getElementById('lbl-ax').textContent=ax;
  document.getElementById('lbl-cor').textContent=cor;
  document.getElementById('lbl-sag').textContent=sag;
  const mm=(v)=>(v*VOXEL).toFixed(1)+'mm';
  document.getElementById('status').textContent=
    `z=${ax} (${mm(ax)})  y=${cor} (${mm(cor)})  x=${sag} (${mm(sag)})  |  voxel ${(VOXEL).toFixed(3)}mm`;
  const isSeg=(band==='seg');
  document.getElementById('leg-t').style.display=isSeg?'inline':'none';
  document.getElementById('leg-b').style.display=isSeg?'inline':'none';
  document.getElementById('leg-scale').textContent=
    (band==='ratio')?'color: hi/lo attenuation ratio':
    (band==='anat')? 'dark=bg  orange=tissue  ivory=bone':'grayscale';
}

function setupCanvas(id, onClickFn){
  const cv=document.getElementById(id);
  cv.addEventListener('click', e=>{
    const r=cv.getBoundingClientRect();
    const sx=cv.width/r.width, sy=cv.height/r.height;
    onClickFn(Math.floor((e.clientX-r.left)*sx), Math.floor((e.clientY-r.top)*sy));
  });
}

// sliders
['ax','cor','sag'].forEach(p=>{
  document.getElementById('sl-'+p).addEventListener('input',e=>{
    if(p==='ax') ax=+e.target.value;
    else if(p==='cor') cor=+e.target.value;
    else sag=+e.target.value;
    updateInfo(); render();
  });
});

// band buttons
document.getElementById('band-btns').addEventListener('click',e=>{
  if(!e.target.dataset.band) return;
  band=e.target.dataset.band;
  document.querySelectorAll('#band-btns button').forEach(b=>b.classList.toggle('active',b.dataset.band===band));
  if(!vols[band]){
    document.getElementById('status').textContent='Decompressing '+band+'...';
    setTimeout(()=>{ vols[band]=decompressVol(band); updateInfo(); render(); },10);
    return;
  }
  updateInfo(); render();
});

// canvas click → update crosshair position
setupCanvas('cv-ax', (cx,cy)=>{ sag=cx; cor=cy; syncSliders(); updateInfo(); render(); });
setupCanvas('cv-cor',(cx,cy)=>{ sag=cx; ax=cy;  syncSliders(); updateInfo(); render(); });
setupCanvas('cv-sag',(cx,cy)=>{ cor=cx; ax=cy;  syncSliders(); updateInfo(); render(); });

function syncSliders(){
  document.getElementById('sl-ax').value=ax;
  document.getElementById('sl-cor').value=cor;
  document.getElementById('sl-sag').value=sag;
}

// startup: load anatomy first
document.getElementById('status').textContent='Decompressing anatomy volume...';
setTimeout(()=>{
  vols['anat']=decompressVol('anat');
  document.getElementById('status').textContent='Ready — click any panel to navigate';
  updateInfo(); render();
},10);
</script>
</body>
</html>
"""

html = HTML.replace('DATA_PLACEHOLDER', data_js)
with open(OUT, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"Wrote {OUT} ({os.path.getsize(OUT)//1024} KB)", flush=True)
