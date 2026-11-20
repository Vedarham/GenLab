'use strict';
/* ═══════════════ UI building blocks ═══════════════ */
const APP = { page: 'home', mode: 'easy', ds: 'digits' };
const store = { get(k, d) { try { return localStorage.getItem('genlab.' + k) || d; } catch (_) { return d; } }, set(k, v) { try { localStorage.setItem('genlab.' + k, v); } catch (_) { } } };
function toast(msg) { const t = h('div.toast', { role: 'status' }, msg); document.body.append(t); setTimeout(() => t.remove(), 2400); }
const dsOf = () => _dsCache[APP.ds];
const T = (a, shape) => tf.tensor(a, shape);

/* parameter controls */
function ParamControls(runner, { advanced = true, only } = {}) {
  const root = h('div'), adv = h('details.adv', null, h('summary', null, 'More settings')), advBox = h('div');
  adv.append(advBox);
  runner.tool.params.forEach(p => {
    if (only && !only(p)) return;
    const val = h('b', null, ''), show = () => val.textContent = p.type === 'select' ? '' : String(runner.params[p.id]);
    let input;
    if (p.type === 'range') {
      input = h('input', { type: 'range', min: p.min, max: p.max, step: p.step, value: runner.params[p.id], id: 'p-' + p.id, oninput: e => { runner.params[p.id] = +e.target.value; show(); } });
    } else {
      input = h('select.sel', { id: 'p-' + p.id, onchange: e => { runner.params[p.id] = typeof p.def === 'number' ? +e.target.value : e.target.value; if (p.id === 'arch' || p.id === 'loss') runner.emit(); } },
        p.options.map(o => { const [v, l] = Array.isArray(o) ? o : [o, o]; return h('option', { value: v, selected: String(v) === String(runner.params[p.id]) }, l); }));
    }
    show();
    const box = h('div.ctl', null, h('label', { for: 'p-' + p.id }, p.label, val), input, p.help && h('div.help', null, p.help));
    (p.adv && advanced ? advBox : root).append(box);
  });
  if (advBox.children.length) root.append(adv);
  return root;
}

/* syntax-highlighted code editor */
const KW = new Set('const let var function return if else for while of in new await async break continue true false null undefined typeof throw try catch class this'.split(' '));
function highlight(src) {
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;'); let out = '', last = 0;
  const re = /(\/\/[^\n]*)|('(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")|(\b\d+\.?\d*(?:e-?\d+)?\b)|([A-Za-z_$][\w$]*)(\s*\()?/g; let m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index)); last = re.lastIndex;
    if (m[1]) out += '<span class="tk-c">' + esc(m[1]) + '</span>'; else if (m[2]) out += '<span class="tk-s">' + esc(m[2]) + '</span>'; else if (m[3]) out += '<span class="tk-n">' + m[3] + '</span>';
    else if (KW.has(m[4])) out += '<span class="tk-k">' + m[4] + '</span>' + esc(m[5] || ''); else if (m[5]) out += '<span class="tk-f">' + m[4] + '</span>' + esc(m[5]); else out += esc(m[0]);
  }
  return out + esc(src.slice(last)) + '\n';
}
function Editor(value, { readOnly = false, onRun } = {}) {
  const ta = h('textarea', { spellcheck: false, readOnly, 'aria-label': 'Code editor', value }), pre = h('pre'), gut = h('div.gutter');
  const sync = () => { pre.innerHTML = highlight(ta.value); gut.textContent = ta.value.split('\n').map((_, i) => i + 1).join('\n'); };
  const scroll = () => { pre.scrollTop = ta.scrollTop; pre.scrollLeft = ta.scrollLeft; gut.scrollTop = ta.scrollTop; };
  ta.addEventListener('input', sync); ta.addEventListener('scroll', scroll);
  ta.addEventListener('keydown', e => {
    if (e.key === 'Tab') { e.preventDefault(); const s = ta.selectionStart; ta.setRangeText('  ', s, ta.selectionEnd, 'end'); sync(); }
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); onRun && onRun(); }
  });
  const el = h('div.editor', null, pre, gut, ta); sync();
  return { el, getValue: () => ta.value, setValue(v) { ta.value = v; sync(); },
    jump(line) { ta.scrollTop = Math.max(0, (line - 2) * 20); scroll(); const ls = ta.value.split('\n'); let o = 0; for (let i = 0; i < line - 1; i++) o += ls[i].length + 1; ta.focus(); ta.setSelectionRange(o, o + ls[line - 1].length); } };
}
function stepsOf(code) { return code.split('\n').map((l, i) => { const m = /^\/\/ ── STEP (\d+) · (.*?) ─*$/.exec(l); return m && { n: m[1], title: m[2].trim(), line: i + 1 }; }).filter(Boolean); }

/* training output (galleries + charts + log) */
const CHART_TITLES = { loss: 'Training curves', parts: 'ELBO terms: reconstruction vs KL', adv: 'Discriminator loss' };
function OutputView(runner, { log = false } = {}) {
  const root = h('div.stack'), bar = h('div.bar', null, h('i')), status = h('div.row.muted', { style: { fontSize: '12.5px' } });
  const charts = h('div.stack'), gal = h('div.stack'), con = h('div.console', { hidden: !log, role: 'log' });
  const empty = h('div.empty', null, 'Nothing here yet. Press ', h('b', null, 'Train'), ' and watch the model learn.');
  root.append(h('div.card', null, h('div', { style: { display: 'grid', gap: '6px' } }, status, bar)), empty, charts, gal, con);
  const cc = {}, gc = {}, groups = {};
  function refresh() {
    const r = runner; bar.firstChild.style.width = (r.progress * 100).toFixed(0) + '%';
    const lab = { idle: 'Ready', running: 'Training… ' + (r.progress * 100).toFixed(0) + '%', done: 'Finished', stopped: 'Stopped', error: 'Error' }[r.status];
    status.textContent = ''; status.append(h('b', { style: { color: r.status === 'error' ? 'var(--err)' : 'var(--ink)' } }, lab), r.logs.length && !log ? h('span.faint', null, r.logs[r.logs.length - 1].m) : null);
    empty.hidden = r.status !== 'idle' || r.gallery.length > 0;
    Object.keys(r.charts).forEach(id => {
      if (!cc[id]) { const cv = h('canvas.chart'); cc[id] = { cv, card: h('div.card', null, h('h3', null, CHART_TITLES[id] || id), cv) }; charts.append(cc[id].card); }
      drawChart(cc[id].cv, r.charts[id]);
    });
    r.gallery.forEach(g => {
      const grp = g.group;
      if (!groups[grp]) { groups[grp] = h('div.gal'); gal.append(h('div.card', null, h('h3', null, grp), groups[grp])); }
      let c = gc[g.key];
      if (!c) { c = gc[g.key] = { box: h('div.gcard', null, h('span', null, g.title)) }; groups[grp].append(c.box); }
      if (c.canvas !== g.canvas) { if (c.canvas) c.canvas.replaceWith(g.canvas); else c.box.append(g.canvas); c.canvas = g.canvas; }
      c.box.firstChild.textContent = g.title;
    });
    if (log) { con.textContent = ''; r.logs.forEach(l => con.append(h('div', { class: l.kind }, l.m))); con.hidden = !r.logs.length; con.scrollTop = 1e9; }
    else if (r.status === 'error') { con.hidden = false; con.textContent = ''; r.logs.filter(l => l.kind === 'err').forEach(l => con.append(h('div.err', null, l.m))); }
  }
  const off = runner.on(refresh); refresh();
  return { el: root, destroy: off, refresh };
}

/* dataset strip / inspector */
function DatasetStrip(ds, n = 24, scale = 2) { return gridCanvas(ds.xTrain.slice([0, 0, 0, 0], [n, ds.S, ds.S, ds.C]), n, scale, 1); }
function DatasetPeek(ds) {
  const root = h('div.stack'), N = 40, scale = 4, gap = 2, cell = ds.S * scale + gap;
  const cv = gridCanvas(ds.xTrain.slice([0, 0, 0, 0], [N, ds.S, ds.S, ds.C]), 10, scale, gap), info = h('div'), mat = h('div', { style: { overflow: 'auto' } });
  let ch = 0, curI = 0;
  function inspect(i) {
    curI = i;
    const d = ds.xTrain.slice([i, 0, 0, 0], [1, ds.S, ds.S, ds.C]).dataSync(); const t = h('table.matrix');
    for (let y = 0; y < ds.S; y++) { const tr = h('tr'); for (let x = 0; x < ds.S; x++) { const v = d[(y * ds.S + x) * ds.C + (ds.C === 1 ? 0 : ch)], g = Math.round(v * 255); tr.append(h('td', { style: { background: ds.C === 1 ? `rgb(${g},${g},${g})` : `rgb(${ch === 0 ? g : 0},${ch === 1 ? g : 0},${ch === 2 ? g : 0})`, color: v > .5 ? '#000' : '#fff' } }, v.toFixed(1).replace('0.', '.').replace('1.0', '1'))); } t.append(tr); }
    mat.replaceChildren(t); info.textContent = `sample #${i} · label "${ds.classes[ds.yTrainArr[i]]}" · caption: "${ds.captions.train[i]}"`;
  }
  cv.style.cursor = 'pointer'; cv.title = 'Click an image to see its pixel numbers';
  cv.addEventListener('click', e => { const r = cv.getBoundingClientRect(), k = cv.width / r.width, x = Math.floor((e.clientX - r.left) * k / cell), y = Math.floor((e.clientY - r.top) * k / cell), i = y * 10 + x; if (i < N) inspect(i); });
  const chSel = ds.C === 3 ? h('div.seg', null, ['R', 'G', 'B'].map((c, i) => h('button', { 'aria-pressed': i === 0, onclick: e => { ch = i; [...e.target.parentNode.children].forEach((b, j) => b.setAttribute('aria-pressed', j === i)); inspect(curI); } }, c))) : null;
  inspect(0);
  root.append(h('div.card', null, h('h3', null, ds.name, h('small', null, `${ds.nTrain} train · ${ds.nTest} test · shape [${ds.S},${ds.S},${ds.C}]`)), h('p.muted', { style: { margin: '0 0 10px' } }, ds.blurb), cv),
    h('div.card', null, h('h3', null, 'What the computer sees', h('small', null, 'a picture is a grid of numbers 0 → 1')), info, chSel, mat));
  return root;
}

/* draw / upload pad */
function InputPad(ds, onChange, { label = 'Draw here' } = {}) {
  const S = ds.S, C = ds.C, scale = 12, buf = new Float32Array(S * S * C), cv = h('canvas.pad', { width: S * scale, height: S * scale, 'aria-label': label });
  const g = cv.getContext('2d'); let color = [1, 1, 1], inv = false;
  const draw = () => { const im = g.createImageData(S, S); for (let p = 0; p < S * S; p++) for (let k = 0; k < 3; k++) im.data[p * 4 + k] = 255 * buf[p * C + (C === 1 ? 0 : k)], im.data[p * 4 + 3] = 255;
    const t = document.createElement('canvas'); t.width = t.height = S; t.getContext('2d').putImageData(im, 0, 0); g.imageSmoothingEnabled = false; g.drawImage(t, 0, 0, S * scale, S * scale); };
  const paint = e => { const r = cv.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / r.width * S), y = Math.floor((e.clientY - r.top) / r.height * S);
    [[0, 0, 1], [1, 0, .45], [-1, 0, .45], [0, 1, .45], [0, -1, .45]].forEach(([dx, dy, w]) => { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= S || yy >= S) return; for (let k = 0; k < C; k++) { const i = (yy * S + xx) * C + k; buf[i] = Math.max(buf[i], color[k] * w); } }); draw(); };
  let down = false; cv.addEventListener('pointerdown', e => { down = true; cv.setPointerCapture(e.pointerId); paint(e); }); cv.addEventListener('pointermove', e => down && paint(e)); cv.addEventListener('pointerup', () => { down = false; onChange(); });
  const set = a => { buf.set(a); draw(); onChange(); };
  const colorIn = C === 3 ? h('input', { type: 'color', value: '#ffffff', 'aria-label': 'Brush colour', oninput: e => { const n = parseInt(e.target.value.slice(1), 16); color = [n >> 16 & 255, n >> 8 & 255, n & 255].map(v => v / 255); } }) : null;
  const file = h('input', { type: 'file', accept: 'image/*', hidden: true, onchange: e => { const f = e.target.files[0]; if (!f) return; const im = new Image(); im.onload = () => {
    const c = document.createElement('canvas'); c.width = c.height = S; const x = c.getContext('2d'); const m = Math.min(im.width, im.height); x.drawImage(im, (im.width - m) / 2, (im.height - m) / 2, m, m, 0, 0, S, S);
    const d = x.getImageData(0, 0, S, S).data, a = new Float32Array(S * S * C); for (let p = 0; p < S * S; p++) for (let k = 0; k < C; k++) { let v = C === 1 ? (d[p * 4] * .299 + d[p * 4 + 1] * .587 + d[p * 4 + 2] * .114) / 255 : d[p * 4 + k] / 255; a[p * C + k] = inv ? 1 - v : v; } set(a); URL.revokeObjectURL(im.src); }; im.src = URL.createObjectURL(f); e.target.value = ''; } });
  const el = h('div', { style: { display: 'grid', gap: '8px', justifyItems: 'start' } }, cv,
    h('div.row', null, colorIn, h('button.btn.sm', { onclick: () => set(new Float32Array(S * S * C)) }, 'Clear'),
      h('button.btn.sm', { onclick: () => { const i = Math.floor(Math.random() * ds.nTest); set(ds.xTest.slice([i, 0, 0, 0], [1, S, S, C]).dataSync()); } }, 'Random test image'),
      h('button.btn.sm', { onclick: () => file.click() }, 'Upload photo'), h('label.faint', { style: { fontSize: '12px' } }, h('input', { type: 'checkbox', onchange: e => inv = e.target.checked }), ' invert')), file);
  draw();
  return { el, get: () => tf.tensor4d(buf.slice(), [1, S, S, C]), isBlank: () => buf.every(v => v === 0) };
}

/* helpers for try-it panels */
const showT = (t, cols, scale = 5) => { const c = gridCanvas(t, cols, scale); return c; };
const section = (title, ...kids) => h('div.card', null, h('h3', null, title), h('div', { style: { display: 'grid', gap: '10px' } }, kids));
const rowCtl = (label, el) => h('div.row', null, h('span.muted', { style: { fontSize: '12.5px' } }, label), el);
function slider(min, max, step, val, onInput, id) { const o = h('b.mono', null, val), i = h('input', { type: 'range', min, max, step, value: val, id, oninput: e => { o.textContent = e.target.value; onInput(+e.target.value); } }); return h('span.row', null, i, o); }
function ClassBars(share, classes) { return h('div.cls-bars', null, share.map((s, i) => h('div', null, h('span', null, classes[i]), h('i', { style: { width: (s * 100) + '%' } }), (s * 100).toFixed(0) + '%'))); }

/* latent explorer: sliders · map · interpolate · variations */
function LatentExplorer(cfg) {
  const { Z, decode, ds } = cfg, S = ds.S, C = ds.C, root = h('div', { style: { display: 'grid', gap: '10px' } });
  const tabs = ['Sliders', ...(Z >= 2 ? ['Map'] : []), 'Interpolate', 'Variations'], body = h('div'), seg = h('div.seg');
  const z = new Float32Array(Z), view = h('div'); let cur = tabs[0];
  const dec = zz => { const t = decode(T(Array.from(zz), [1, Z])); const o = t; return o; };
  const pages = {
    Sliders() {
      const out = h('div'), sls = h('div', { style: { display: 'grid', gap: '2px 14px', gridTemplateColumns: 'repeat(auto-fill,minmax(180px,1fr))' } });
      const upd = () => { const t = dec(z), cv = gridCanvas(t, 1, 14, 0); t.dispose(); out.replaceChildren(cv); };
      const ins = [];
      for (let i = 0; i < Math.min(Z, 8); i++) { const inp = h('input', { type: 'range', min: -3, max: 3, step: .1, value: z[i], 'aria-label': 'dimension ' + (i + 1), oninput: e => { z[i] = +e.target.value; upd(); } }); ins.push(inp); sls.append(h('label.muted.mono', { style: { fontSize: '11px' } }, 'z' + (i + 1), inp)); }
      upd();
      return h('div.row', { style: { alignItems: 'flex-start' } }, out, h('div', { style: { flex: 1, minWidth: '200px' } }, sls, Z > 8 && h('div.faint', { style: { fontSize: '12px' } }, 'Other dimensions stay at 0.'),
        h('div.row', { style: { marginTop: '8px' } }, h('button.btn.sm', { onclick: () => { for (let i = 0; i < Z; i++) z[i] = (Math.random() * 2 - 1) * 2; ins.forEach((s, i) => s.value = z[i]); upd(); } }, 'Randomise'), h('button.btn.sm', { onclick: () => { z.fill(0); ins.forEach(s => s.value = 0); upd(); } }, 'Centre'))));
    },
    Map() {
      let a = 0, b = 1; const out = h('div'), n = 11;
      const draw = () => { const zs = []; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const v = new Float32Array(Z); v[a] = (c / (n - 1) * 2 - 1) * 2.5; v[b] = (r / (n - 1) * 2 - 1) * 2.5; zs.push(...v); } const t = decode(T(zs, [n * n, Z])), cv = gridCanvas(t, n, 4, 1); t.dispose(); out.replaceChildren(cv); };
      const opts = i => Array.from({ length: Z }, (_, k) => h('option', { value: k, selected: k === i }, 'z' + (k + 1)));
      draw(); return h('div', { style: { display: 'grid', gap: '8px' } }, h('div.row', null, Z > 2 && [h('span.muted', null, 'x axis'), h('select.sel', { onchange: e => { a = +e.target.value; draw(); } }, opts(0)), h('span.muted', null, 'y axis'), h('select.sel', { onchange: e => { b = +e.target.value; draw(); } }, opts(1))], h('span.faint', null, 'each cell decodes one point of the grid; the rest of z stays at 0')), out);
    },
    Interpolate() {
      const out = h('div'), go = () => { const [a, b] = cfg.pair(), zs = []; for (let i = 0; i < 10; i++) { const t = i / 9; for (let k = 0; k < Z; k++) zs.push(a[k] * (1 - t) + b[k] * t); } const im = decode(T(zs, [10, Z])); out.replaceChildren(gridCanvas(im, 10, 5, 2)); im.dispose(); };
      go(); return h('div', { style: { display: 'grid', gap: '8px' } }, h('div.row', null, h('button.btn.sm', { onclick: go }, 'New pair'), h('span.faint', null, cfg.encode ? 'two real test images, encoded, then blended in latent space' : 'two random noise vectors blended')), out);
    },
    Variations() {
      let sigma = .5, base; const out = h('div'), go = () => { base = cfg.base(); draw(); }, draw = () => { const zs = []; for (let i = 0; i < 24; i++) for (let k = 0; k < Z; k++) zs.push(base[k] + (i ? (Math.random() * 2 - 1) * 1.7 * sigma : 0)); const im = decode(T(zs, [24, Z])); out.replaceChildren(gridCanvas(im, 12, 4, 2)); im.dispose(); };
      go(); return h('div', { style: { display: 'grid', gap: '8px' } }, h('div.row', null, h('button.btn.sm', { onclick: go }, 'New starting point'), rowCtl('variation', slider(0.1, 2, 0.1, .5, v => { sigma = v; draw(); }))), h('span.faint', null, 'First image is the starting point; the rest are nearby points in latent space.'), out);
    },
  };
  const show = t => { cur = t; [...seg.children].forEach(b => b.setAttribute('aria-pressed', b.textContent === t)); view.replaceChildren(pages[t]()); };
  tabs.forEach(t => seg.append(h('button', { 'aria-pressed': t === cur, onclick: () => show(t) }, t)));
  root.append(seg, view); show(cur); return root;
}
