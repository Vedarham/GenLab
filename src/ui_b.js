'use strict';
/* ═══════════════ "Try it" panels, pages and shell ═══════════════ */
const TRYIT = {};
const testBatch = (ds, n) => { const idx = Array.from({ length: n }, () => Math.floor(Math.random() * ds.nTest)); return tf.gather(ds.xTest, tf.tensor1d(idx, 'int32')); };
const mseOf = (a, b) => tf.tidy(() => tf.mean(tf.square(tf.sub(a, b))).dataSync()[0]);

function reconPad(ds, fn, labels, extra) {   // draw/upload → rows of images
  const out = h('div'), pad = InputPad(ds, () => run());
  const run = () => { if (pad.isBlank()) { out.replaceChildren(h('span.faint', null, 'Draw something or press “Random test image”.')); return; }
    const x = pad.get(), ys = fn(x), rows = tf.concat([x, ...ys]); out.replaceChildren(gridCanvas(rows, rows.shape[0], 12, 3), h('div.faint.mono', { style: { fontSize: '11.5px', marginTop: '4px' } }, labels.join('  →  '))); };
  run(); return h('div.row', { style: { alignItems: 'flex-start', gap: '18px' } }, pad.el, out);
}
TRYIT.ae = (R, host) => {
  const ds = dsOf(), api = R.api, out = h('div'), info = h('div.faint.mono');
  const go = () => { const x = testBatch(ds, 8), r = api.reconstruct(x), both = tf.concat([x, r]); out.replaceChildren(gridCanvas(both, 8, 6, 3)); info.textContent = `mean squared error ${mseOf(x, r).toFixed(4)} · code size ${api.latent} numbers vs ${ds.D} pixels (${(ds.D / api.latent).toFixed(1)}× compression)`; x.dispose(); r.dispose(); both.dispose(); };
  go();
  host.append(section('Reconstruct test images', h('div.row', null, h('button.btn.sm', { onclick: go }, 'New images'), h('span.faint', null, 'top: original · bottom: rebuilt')), out, info),
    section('Reconstruct your own image', reconPad(ds, x => [api.reconstruct(x)], ['your input', 'rebuilt'])));
};
TRYIT.dae = (R, host) => {
  const ds = dsOf(), api = R.api; let level = R.params.noise;
  const out = h('div'), go = () => { const x = testBatch(ds, 8), n = api.corrupt(x, level), d = api.denoise(n), all = tf.concat([x, n, d]); out.replaceChildren(gridCanvas(all, 8, 6, 3)); x.dispose(); n.dispose(); d.dispose(); all.dispose(); };
  go();
  host.append(section('Clean test images', h('div.row', null, h('button.btn.sm', { onclick: go }, 'New images'), rowCtl('damage level', slider(0.05, 1, 0.05, level, v => { level = v; go(); }))), h('span.faint', null, 'rows: clean · damaged · repaired'), out),
    section('Repair your own image', reconPad(ds, x => { const n = api.corrupt(x, level); return [n, api.denoise(n)]; }, ['your input', 'damaged', 'repaired'])));
};
function generateGrid(sampleFn, cols = 8, n = 24, scale = 5) {
  const out = h('div'), go = () => { const t = sampleFn(n); out.replaceChildren(gridCanvas(t, cols, scale, 3)); t.dispose(); }; go();
  return { out, go };
}
TRYIT.vae = (R, host) => {
  const ds = dsOf(), api = R.api, g = generateGrid(api.sample);
  host.append(section('Generate new images', h('button.btn.sm', { onclick: g.go }, 'Sample again'), g.out),
    section('Explore the latent space', LatentExplorer({ Z: api.Z, decode: api.decode, encode: true, ds,
      pair() { const x = testBatch(ds, 2), { mu } = api.encode(x), a = mu.dataSync(), r = [a.slice(0, api.Z), a.slice(api.Z)]; x.dispose(); mu.dispose(); return r; },
      base() { const x = testBatch(ds, 1), { mu } = api.encode(x), a = Array.from(mu.dataSync()); x.dispose(); mu.dispose(); return a; } })),
    section('Encode your own image', reconPad(ds, x => { const { mu, logvar } = api.encode(x); const r = api.decode(mu); mu.dispose(); logvar.dispose(); return [r]; }, ['your input', 'decode(μ)'])));
};
async function diversityPanel(R, api) {
  const ds = dsOf(), box = h('div'), btn = h('button.btn.sm', { onclick: async () => {
    btn.disabled = true; box.textContent = 'Classifying 300 generated images…'; const ev = await getEvaluator(ds), s = api.sample(300), f = fitToDataset(s, ds), { P } = await ev.features(f); s.dispose(); f.dispose();
    const st = METRICS.classStats(P), bad = st.covered < ds.K * .7 || st.entropy < .75;
    box.replaceChildren(ClassBars(st.share, ds.classes), h('p', { class: bad ? 'warn' : 'good', style: { margin: '8px 0 0' } }, `${st.covered}/${st.K} classes present · entropy ${(st.entropy * 100).toFixed(0)}% of maximum. ` + (bad ? 'Some modes are missing: possible mode collapse.' : 'Healthy variety.'))); btn.disabled = false; } }, 'Check diversity');
  return section('Mode-collapse check', h('div.row', null, btn, h('span.faint', null, 'A classifier labels 300 samples; a collapsed GAN covers only a few classes.')), box);
}
TRYIT.gan = (R, host) => {
  const ds = dsOf(), api = R.api, g = generateGrid(api.sample);
  host.append(section('Generate (' + api.name + ')', h('button.btn.sm', { onclick: g.go }, 'Sample again'), g.out),
    section('Explore the noise space', LatentExplorer({ Z: api.Z, decode: api.decode, ds, pair: () => [Float32Array.from({ length: api.Z }, () => tf.util.randNormal ? Math.random() * 2 - 1 : 0), Float32Array.from({ length: api.Z }, () => Math.random() * 2 - 1)], base: () => Array.from({ length: api.Z }, () => (Math.random() * 2 - 1) * 1.2) })));
  diversityPanel(R, api).then(p => host.append(p));
};
TRYIT.cyclegan = (R, host) => {
  const ds = dsOf(), api = R.api, out = h('div'); let dir = 'ab';
  const go = () => { const x = dir === 'ab' ? api.aTest : api.bTest, y = dir === 'ab' ? api.toStyle(x) : api.fromStyle(x), back = dir === 'ab' ? api.fromStyle(y) : api.toStyle(y), rows = tf.concat([x, y, back]); out.replaceChildren(gridCanvas(rows, 8, 6, 3)); y.dispose(); back.dispose(); rows.dispose(); };
  go();
  const seg = h('div.seg', null, [['ab', 'Original → ' + STYLES[R.params.style].label.split(' (')[0]], ['ba', 'Styled → Original']].map(([k, l]) => h('button', { 'aria-pressed': k === dir, onclick: e => { dir = k; [...seg.children].forEach(b => b.setAttribute('aria-pressed', b === e.target)); go(); } }, l)));
  const usePad = () => reconPad(ds, x => { const y = api.toStyle(x); return [y, api.fromStyle(y)]; }, ['your input', 'G(input) restyled', 'F(G(input)) cycled back']);
  host.append(section('Translate test images', seg, h('span.faint', null, 'rows: input · translated · translated back (the cycle)'), out), section('Restyle your own image', usePad()));
};
TRYIT.diffusion = (R, host) => {
  const ds = dsOf(), api = R.api, out = h('div'); let cls = -1, w = 2;
  const go = () => { const classes = cls < 0 ? Array.from({ length: ds.K }, (_, i) => i) : [cls], { images, traj } = api.sample(4, classes, w);
    const rows = []; for (let i = 0; i < 4; i++) traj.forEach(t => rows.push(t.slice([i, 0, 0, 0], [1, ds.S, ds.S, ds.C])));
    const strip = tf.concat([...rows, ...Array(0)]); out.replaceChildren(h('span.faint', null, 'Each row is one image: left = pure noise, right = finished.'), gridCanvas(strip, traj.length, 6, 3));
    images.dispose(); traj.forEach(t => t.dispose()); rows.forEach(r => r.dispose()); strip.dispose(); };
  host.append(section('Watch the denoising', h('div.row', null,
    h('select.sel', { 'aria-label': 'class', onchange: e => cls = +e.target.value }, h('option', { value: -1 }, 'Any class'), ds.classes.map((c, i) => h('option', { value: i }, c))),
    rowCtl('guidance', slider(0, 6, 0.5, w, v => w = v)), h('button.btn.primary.sm', { onclick: go }, 'Generate')), out));
  go();
};
TRYIT.text2img = (R, host) => {
  const ds = dsOf(), api = R.api, out = h('div'), msg = h('p.muted', { style: { margin: 0 } }); let w = 2.5;
  const input = h('input.sel', { id: 'prompt', type: 'text', style: { flex: 1, minWidth: '200px', padding: '8px 10px' }, value: PROMPT_HINTS[ds.id][0], 'aria-label': 'Prompt', onkeydown: e => e.key === 'Enter' && go() });
  const go = () => { const cls = textToClasses(ds.id, input.value); if (!cls) { msg.textContent = 'No words I know in that prompt. Try one of the suggestions below.'; return; }
    msg.textContent = 'Understood as: ' + cls.map(c => ds.classes[c]).join(' + '); const { images, traj } = api.sample(16, cls, w); out.replaceChildren(gridCanvas(images, 8, 6, 3)); images.dispose(); traj.forEach(t => t.dispose()); };
  host.append(section('Describe an image', h('div.row', null, input, h('button.btn.primary', { onclick: go }, 'Generate')), h('div.row', null, h('span.faint', null, 'Try:'), PROMPT_HINTS[ds.id].map(p => h('span.chip', { onclick: () => { input.value = p; go(); } }, p))), rowCtl('guidance (prompt strength)', slider(0, 6, 0.5, w, v => w = v)), msg, out));
  go();
};
TRYIT.realnvp = (R, host) => {
  let temp = .7; const ds = dsOf(), api = R.api, g = generateGrid(n => api.sample(n, temp));
  const ex = h('div'), exact = () => { const x = testBatch(ds, 8), z = api.toZ(x), r = api.fromZ(z), all = tf.concat([x, r]); ex.replaceChildren(gridCanvas(all, 8, 6, 3), h('div.faint.mono', { style: { fontSize: '11.5px' } }, 'max pixel error ' + tf.tidy(() => tf.abs(x.sub(r)).max().dataSync()[0]).toExponential(1) + ' (flows are exactly invertible)')); x.dispose(); z.dispose(); r.dispose(); all.dispose(); };
  exact();
  host.append(section('Sample', h('div.row', null, h('button.btn.sm', { onclick: g.go }, 'Sample again'), rowCtl('temperature', slider(0.2, 1.2, 0.1, temp, v => temp = v))), g.out, h('span.faint.mono', null, 'test likelihood ≈ ' + api.bitsPerDim().toFixed(2) + ' bits/dim (lower is better)')),
    section('Image → noise → image', h('button.btn.sm', { onclick: exact }, 'New images'), h('span.faint', null, 'top: original · bottom: forward to noise z, then backward'), ex));
};
TRYIT.pixelrnn = (R, host) => {
  const ds = dsOf(), api = R.api, g = generateGrid(n => api.sample(n), 8, 16, 8), out = h('div'); let rows = 3;
  const go = () => { const x = testBatch(ds, 1), bits = api.toBits(x), masked = tf.tidy(() => { const m = tf.concat([tf.ones([1, rows, api.R, 1]), tf.zeros([1, api.R - rows, api.R, 1])], 1); return bits.mul(m).add(tf.sub(1, m).mul(0.5)); }), comp = api.complete(bits, rows, 6), all = tf.concat([bits, masked, comp]);
    out.replaceChildren(gridCanvas(all, 8, 8, 3)); x.dispose(); bits.dispose(); masked.dispose(); comp.dispose(); all.dispose(); };
  go();
  host.append(section('Draw from scratch', h('button.btn.sm', { onclick: g.go }, 'Sample again'), g.out),
    section('Complete the image', h('div.row', null, h('button.btn.sm', { onclick: go }, 'New image'), rowCtl('rows given', slider(1, 7, 1, rows, v => { rows = v; go(); }))), h('span.faint', null, '1st: original · 2nd: what the model sees (gray = unknown) · rest: six completions'), out));
};
TRYIT.captioner = (R, host) => {
  const ds = dsOf(), api = R.api, list = h('div', { style: { display: 'grid', gap: '8px' } }); let temp = .6;
  const go = () => { const x = testBatch(ds, 6), caps = api.caption(x, temp); list.replaceChildren(...caps.map((c, i) => h('div.row', { style: { gap: '12px' } }, gridCanvas(x.slice([i, 0, 0, 0], [1, ds.S, ds.S, ds.C]), 1, 6, 0), h('div', null, h('div', null, h('b', null, '“' + c + '”')))))); x.dispose(); };
  go();
  const out = h('div'), pad = InputPad(ds, () => { if (pad.isBlank()) return; const x = pad.get(); out.textContent = '“' + api.caption(x, temp)[0] + '”'; x.dispose(); });
  host.append(section('Captions for test images', h('div.row', null, h('button.btn.sm', { onclick: go }, 'New images'), rowCtl('temperature', slider(0.2, 1.8, 0.1, temp, v => temp = v))), list),
    section('Caption your own image', h('div.row', { style: { alignItems: 'flex-start', gap: '18px' } }, pad.el, h('b', { style: { fontSize: '16px' } }, out))));
};

/* ───────── pages ───────── */
const main = () => $('.main');
function StatusDot(runner) { return runner.api ? h('i.dot', { title: 'trained' }) : null; }

function ToolPage(tool) {
  const R = getRunner(tool, APP.ds), ds = dsOf(), root = h('div'), learn = APP.mode === 'learn';
  root.append(h('div.tool-head', null, h('div', null, h('h2', null, tool.name), h('p', null, tool.tagline)), h('span.pill', null, tool.syllabus)));
  const tryHost = h('div.stack');
  const unsub = R.on(() => { if (R.api && tryHost._api !== R.api) { tryHost._api = R.api; tryHost.replaceChildren(); try { TRYIT[tool.id](R, tryHost); } catch (e) { console.error(e); tryHost.append(h('div.console', null, 'Could not build the Try-it panel: ' + e.message + '. If you edited the code, keep the returned object the same shape.')); } if (learn) tabs && tabs.mark('Try it'); } if (learn && R.status === 'running') tabs && tabs.select('Output'); });
  let tabs = null;
  const concept = h('div.card.prose', { html: tool.concept }), tries = h('div.card', null, h('h3', null, 'Things to try'), h('ul', { style: { margin: 0, paddingLeft: '18px' } }, tool.tries.map(t => h('li', null, t))));
  const trainBtn = h('button.btn.primary' + (learn ? '' : '.big'), { onclick: () => R.status === 'running' ? (R.stop = true) : (R.code = learn ? edit.getValue() : RECIPES[R.tool.recipe], tryHost._api = null, R.run()) }, 'Train');
  const syncBtn = () => { trainBtn.textContent = R.status === 'running' ? 'Stop' : R.api ? (learn ? '▶ Run again' : 'Train again') : learn ? '▶ Run' : 'Train'; };
  R.on(syncBtn); syncBtn();
  const out = OutputView(R, { log: learn });
  if (R.api) { tryHost._api = R.api; try { TRYIT[tool.id](R, tryHost); } catch (e) { console.error(e); } }
  let edit;
  if (!learn) {
    const presets = h('div.seg', { style: { width: '100%' } }), box = h('div'), setP = k => { Object.assign(R.params, R.tool.presets[k]); [...presets.children].forEach(b => b.setAttribute('aria-pressed', b.dataset.k === k)); build(); };
    const build = () => box.replaceChildren(ParamControls(R, { only: p => !tool.shared || true }));
    Object.keys(R.tool.presets).forEach(k => presets.append(h('button', { 'data-k': k, 'aria-pressed': k === 'balanced', style: { flex: 1 }, onclick: () => setP(k) }, { quick: 'Quick', balanced: 'Balanced', thorough: 'Thorough' }[k])));
    build(); R.on(() => { if (tool.id === 'gan') { /* arch/loss label may change */ } });
    const left = h('div.card', null, h('h3', null, 'Dataset', h('small', null, ds.name)), DatasetStrip(ds, 24, 2).constructor === HTMLCanvasElement ? h('div.ds-strip', null, DatasetStrip(ds, 24, 2)) : null, h('p.faint', { style: { margin: '0 0 12px', fontSize: '12.5px' } }, ds.blurb + ' Change dataset at the top.'),
      h('h3', { style: { fontSize: '13px' } }, 'Training length'), presets, h('div', { style: { height: '10px' } }), box, trainBtn, tool.shared ? h('p.faint', { style: { fontSize: '12px' } }, 'Shares the model with Diffusion Lab.') : null);
    root.append(h('div.easy', null, left, h('div.stack', null, out.el, tryHost, h('details.card', null, h('summary', { style: { cursor: 'pointer' } }, 'How does it work?'), h('div.prose', { html: tool.concept }), tries))));
  } else {
    edit = Editor(R.code, { onRun: () => trainBtn.click() });
    const steps = stepsOf(R.code), chipBox = h('div.chips', null, h('span.faint', { style: { fontSize: '12px', alignSelf: 'center' } }, 'Jump to:'), steps.map(s => h('span.chip', { onclick: () => edit.jump(s.line) }, s.n + ' · ' + s.title)));
    const params = h('div.params'); params.append(ParamControls(R, { advanced: false }));
    const leftCard = h('div.card', null, h('div.bar-row', null, trainBtn, h('button.btn.sm', { onclick: () => { edit.setValue(RECIPES[R.tool.recipe]); toast('Original code restored'); } }, 'Reset code'), h('span.faint', { style: { fontSize: '12px' } }, 'Edit anything, then Run (Ctrl+Enter)')), chipBox, params, edit.el);
    const panes = { Data: () => DatasetPeek(ds), Output: () => out.el, 'Try it': () => tryHost, Concept: () => h('div.stack', null, concept, tries) };
    const body = h('div.tabbody'), bar = h('div.tabs', { role: 'tablist' });
    const sel = n => { [...bar.children].forEach(b => b.setAttribute('aria-selected', b.dataset.n === n)); body.replaceChildren(panes[n]()); const b = [...bar.children].find(b => b.dataset.n === n); b.querySelector('i') && b.querySelector('i').remove(); };
    Object.keys(panes).forEach(n => bar.append(h('button', { role: 'tab', 'data-n': n, onclick: () => sel(n) }, n)));
    tabs = { select: sel, mark: n => { const b = [...bar.children].find(b => b.dataset.n === n); if (b && b.getAttribute('aria-selected') !== 'true' && !b.querySelector('i')) b.append(h('i', { style: { display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', background: 'var(--ok)', marginLeft: '6px' } })); } };
    sel(R.api ? 'Try it' : 'Data');
    root.append(h('div.learn', null, leftCard, h('div.card', null, bar, body)));
  }
  root._cleanup = () => { unsub(); out.destroy(); };
  return root;
}

async function ComparePage() {
  const ds = dsOf(), learn = APP.mode === 'learn', root = h('div');
  root.append(h('div.tool-head', null, h('div', null, h('h2', null, 'Compare & Metrics'), h('p', null, 'Score every generator you trained on quality (FID, IS) and variety (diversity, class coverage, mode-collapse check).')), h('span.pill', null, 'FID · IS · Mode collapse')));
  const entries = () => Object.entries(ZOO).filter(([k]) => k.startsWith(ds.id + ':')), list = h('div.stack'), result = h('div.stack'), prog = h('div.muted');
  const quick = ['vae', 'gan', 'diffusion', 'realnvp'];
  async function trainAll() {
    btnAll.disabled = true;
    const jobs = [['vae', {}], ['gan', { loss: 'gan', arch: 'mlp' }], ['gan', { loss: 'wgan', arch: 'mlp' }], ['diffusion', {}], ['realnvp', {}]];
    for (const [id, over] of jobs) { const t = TOOLS_BY_ID[id], R = getRunner(t, ds.id); Object.assign(R.params, t.presets.quick, over); R.code = RECIPES[t.recipe]; prog.textContent = 'Training ' + t.name + ' ' + (over.loss || '') + '…'; const off = R.on(() => prog.textContent = `Training ${t.name} ${over.loss || ''}… ${(R.progress * 100).toFixed(0)}%`); await R.run(); off(); }
    prog.textContent = 'All quick models trained.'; btnAll.disabled = false; renderList();
  }
  const btnAll = h('button.btn', { onclick: trainAll }, 'Quick-train 5 generators'), btnEval = h('button.btn.primary', { onclick: evaluate }, 'Run evaluation');
  function renderList() {
    const es = entries(); list.replaceChildren(h('div.card', null, h('h3', null, 'Trained generators on ' + ds.name), es.length ? h('div.row', null, es.map(([k, v]) => h('span.pill', null, v.label))) : h('p.muted', { style: { margin: '0 0 8px' } }, 'None yet. Train models in their tabs, or use the shortcut:'), h('div.row', { style: { marginTop: '10px' } }, btnAll, btnEval, prog)));
  }
  async function evaluate() {
    const es = entries(); if (!es.length) { toast('Train at least one generator first'); return; } btnEval.disabled = true; result.replaceChildren(h('div.empty', null, 'Evaluating… (the first run trains a small classifier)'));
    try {
      const ev = await getEvaluator(ds), real = await ev.features(ds.xTest), train = await ev.features(ds.xTrain.slice([0, 0, 0, 0], [400, ds.S, ds.S, ds.C])), rs = METRICS.meanCov(real.F);
      const rows = [{ label: 'Real data (reference)', fid: METRICS.fid(rs, METRICS.meanCov(train.F)), is: METRICS.inceptionScore(real.P), div: METRICS.diversity(real.flat), st: METRICS.classStats(real.P), real: true }], thumbs = [];
      for (const [, e] of es) { const raw = await e.sample(300), x = fitToDataset(raw, ds), f = await ev.features(x), st = METRICS.classStats(f.P); rows.push({ label: e.label, fid: METRICS.fid(rs, METRICS.meanCov(f.F)), is: METRICS.inceptionScore(f.P), div: METRICS.diversity(f.flat), st }); thumbs.push([e.label, gridCanvas(x.slice([0, 0, 0, 0], [8, ds.S, ds.S, ds.C]), 8, 3, 1)]); raw.dispose(); x.dispose(); }
      const rd = rows[0].div, maxF = Math.max(...rows.slice(1).map(r => r.fid), 1e-6), gens = rows.slice(1), best = gens.length ? gens.reduce((a, b) => a.fid < b.fid ? a : b) : null;
      const flag = r => r.real ? '' : (r.st.covered < r.st.K * .7 || r.st.entropy < .75) ? 'possible mode collapse' : r.div < rd * .6 ? 'low diversity' : 'ok';
      result.replaceChildren(
        h('div.card', null, h('h3', null, 'Scoreboard', h('small', null, `evaluator accuracy ${(ev.acc * 100).toFixed(0)}% · 300 samples per model`)),
          h('div', { style: { overflow: 'auto' } }, h('table.t', null, h('thead', null, h('tr', null, ['Model', 'FID ↓', 'IS ↑', 'Diversity', 'Classes', 'Entropy', 'Verdict'].map(c => h('th', null, c)))),
            h('tbody', null, rows.map(r => h('tr', null, h('td', null, r.label), h('td', null, r.fid.toFixed(2)), h('td', null, r.is.toFixed(2)), h('td', null, (r.div / rd * 100).toFixed(0) + '%'), h('td', null, r.st.covered + '/' + r.st.K), h('td', null, (r.st.entropy * 100).toFixed(0) + '%'), h('td', { class: flag(r) === 'ok' ? 'good' : 'warn' }, flag(r) || 'reference')))))),
          best && h('p', { style: { margin: '10px 0 0' } }, h('b', null, best.label), ' has the lowest FID (closest to real data). ', rows.filter(r => !r.real && flag(r) !== 'ok').map(r => r.label + ': ' + flag(r) + '. ').join(''))),
        h('div.card', null, h('h3', null, 'FID by model', h('small', null, 'shorter bar = closer to real images')), h('div.cls-bars', null, gens.map(r => h('div', { style: { gridTemplateColumns: '150px 1fr 60px' } }, h('span', null, r.label), h('i', { style: { width: (r.fid / maxF * 100) + '%', background: r === best ? 'var(--s3)' : 'var(--s1)' } }), r.fid.toFixed(1))))),
        h('div.card', null, h('h3', null, 'Samples'), h('div.gal', null, thumbs.map(([l, c]) => h('div.gcard', null, h('span', null, l), c)))),
        h('div.card', null, h('h3', null, 'Class coverage per model', h('small', null, 'what the evaluator thinks each model drew')), h('div.grid2', null, gens.map(r => h('div', null, h('div', { style: { fontWeight: 600, marginBottom: '4px' } }, r.label), ClassBars(r.st.share, ds.classes))))));
    } catch (e) { console.error(e); result.replaceChildren(h('div.console', null, 'Evaluation failed: ' + e.message)); }
    btnEval.disabled = false;
  }
  renderList(); root.append(list, result);
  root.append(h('div.card.prose', { style: { marginTop: '14px' }, html: `<h3>How to read these numbers</h3><p><b>FID</b> compares the feature statistics of generated and real images (lower is better; 0 = identical). <b>IS</b> rewards images that a classifier is sure about <em>and</em> that cover many classes (higher is better; the maximum is the number of classes). <b>Diversity</b> is the typical pixel distance between two samples relative to real data. <b>Classes / Entropy</b> expose <b>mode collapse</b>: if a GAN only draws two kinds of digit, class coverage drops even when each image looks fine.</p><p class="faint">Because these images are tiny, the features come from a small classifier trained on this dataset rather than Inception-v3. The formulas are identical.</p>` }));
  if (learn) root.append(h('div.card', null, h('h3', null, 'The metrics code', h('small', null, 'this exact source is what runs')), h('div', { style: { height: '420px', display: 'flex', flexDirection: 'column', border: '1px solid var(--line)', borderRadius: '8px', overflow: 'hidden' } }, Object.assign(Editor(METRICS_SRC, { readOnly: true }).el, { style: 'flex:1' }))));
  return root;
}

const SYLLABUS = [
  ['Autoencoders', 'Image reconstruction with a bottleneck', 'ae'], ['Regularization', 'Dropout, L2 and training-set size with train-vs-validation curves', 'ae'], ['Denoising autoencoders', 'Repair noisy, masked or salt-and-pepper images', 'dae'],
  ['VAE and ELBO', 'Face and digit generation, reparameterisation, β-VAE, live ELBO terms', 'vae'], ['Latent space exploration', 'Sliders, 2-D map, interpolation, variations', 'vae'], ['GAN and DCGAN', 'MLP and convolutional generators, learning timeline', 'gan'],
  ['WGAN', 'Clipped critic, compared against vanilla GAN', 'gan'], ['Mode collapse', 'Class-coverage and entropy check', 'compare'], ['CycleGAN', 'Unpaired style transfer with cycle-consistency loss', 'cyclegan'],
  ['Diffusion models', 'DDPM with classifier-free guidance, denoising trajectory', 'diffusion'], ['Text-to-image + Transformers', 'Prompt parser conditions the diffusion model; GPT-style captioner', 'text2img'],
  ['RealNVP', 'Affine coupling flow with exact likelihood and invertibility demo', 'realnvp'], ['PixelRNN', 'GRU autoregressive generation and image completion', 'pixelrnn'], ['Transformers / GPT', 'Causal self-attention captioner', 'captioner'], ['FID and IS', 'Mini-FID and Inception-style score with the formulas shown', 'compare'],
];
function HomePage() {
  const root = h('div.hero');
  root.append(h('div', null, h('h1', null, 'GenLab'), h('p.muted', { style: { fontSize: '17px', maxWidth: '60ch', margin: '8px 0 0' } }, 'A studio for generative AI that trains real models in your browser. Pick a dataset, train, and play with what the model made. Switch to Learn Lab to read and edit the code that built it.')),
    h('div.grid2', null,
      h('div.card', null, h('h3', null, 'Easy Studio'), h('p.muted', null, 'Choose a dataset, press Train, then use the model: reconstruct a drawing, slide through latent space, type a prompt.'), h('button.btn.primary', { onclick: () => { setMode('easy'); go('vae'); } }, 'Start with a VAE')),
      h('div.card', null, h('h3', null, 'Learn Lab'), h('p.muted', null, 'Template code on the left, dataset and results on the right. Every step is labelled; edit the code, run it, see what changes.'), h('button.btn', { onclick: () => { setMode('learn'); go('ae'); } }, 'Open the autoencoder code'))));
  NAV.forEach(([g, ids]) => root.append(h('div', null, h('h4', { style: { font: '600 11px var(--font-m)', letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--ink-3)', margin: '8px 0' } }, g), h('div.tiles', null, ids.map(id => { const t = TOOLS_BY_ID[id]; return h('button.tile', { onclick: () => go(id) }, h('span.g', null, t.glyph), h('h3', null, t.name), h('p', null, t.tagline)); })))));
  root.append(h('div', null, h('h4', { style: { font: '600 11px var(--font-m)', letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--ink-3)', margin: '8px 0' } }, 'Evaluate'), h('div.tiles', null, h('button.tile', { onclick: () => go('compare') }, h('span.g', null, 'FID'), h('h3', null, 'Compare & Metrics'), h('p', null, 'FID, Inception Score, diversity and mode-collapse analysis.')), h('button.tile', { onclick: () => go('map') }, h('span.g', null, 'MAP'), h('h3', null, 'Project Map'), h('p', null, 'Syllabus topic → where it lives in the app.')))));
  return root;
}
function MapPage() {
  const root = h('div'); root.append(h('div.tool-head', null, h('div', null, h('h2', null, 'GenLab Project Map'), h('p', null, 'A Unified Generative AI Studio for Image Synthesis, Reconstruction and Evaluation.'))));
  root.append(h('div.card', null, h('div', { style: { overflow: 'auto' } }, h('table.t', null, h('thead', null, h('tr', null, ['Syllabus topic', 'Implementation in GenLab', 'Open'].map(c => h('th', null, c)))), h('tbody', null, SYLLABUS.map(([a, b, id]) => h('tr', null, h('td', null, a), h('td', { style: { textAlign: 'left' } }, b), h('td', null, h('button.btn.sm', { onclick: () => go(id) }, 'Open')))))))),
    h('div.card.prose', { style: { marginTop: '14px' }, html: `<h3>Architecture</h3><p><b>Datasets</b> are drawn procedurally on a canvas (digits, shapes, cartoon faces at 16×16) so the app works offline and trains in seconds. <b>Recipes</b> are plain TensorFlow.js scripts, one per model; the same script is shown in Learn Lab and run by Easy Studio. A <b>Runner</b> executes a recipe, streams losses and image grids to the screen and keeps the trained model so the Try-it panels and the Compare tab can use it.</p><h3>Honest limits</h3><p>Models are deliberately small so they train in a browser tab. Images are 16×16, so quality is a teaching demo, not a photo generator. The text encoder is a keyword parser, the PixelRNN works on 8×8 black-and-white pictures, and FID uses a small dataset-specific classifier instead of Inception-v3.</p>` }));
  return root;
}

/* ───────── shell ───────── */
function go(page) { location.hash = page; }
function setMode(m) { APP.mode = m; store.set('mode', m); }
let current = null;
async function render() {
  if (current && current._cleanup) current._cleanup();
  const m = main(); if (!m) return; m.replaceChildren(h('div.empty', null, 'Loading ' + DATASETS[APP.ds].name + '…'));
  const ds = await loadDataset(APP.ds); if (!ds) return;
  let el;
  try { el = APP.page === 'home' ? HomePage() : APP.page === 'map' ? MapPage() : APP.page === 'compare' ? await ComparePage() : ToolPage(TOOLS_BY_ID[APP.page]); }
  catch (e) { console.error(e); el = h('div.console', null, 'Could not render: ' + e.message); }
  current = el; m.replaceChildren(el); m.scrollTop = 0; renderNav();
}
function renderNav() {
  const nav = $('.nav'); nav.replaceChildren(h('button', { 'aria-current': APP.page === 'home' ? 'page' : null, onclick: () => go('home') }, h('span.g', null, 'GL'), 'Home'));
  NAV.forEach(([g, ids]) => { nav.append(h('h4', null, g)); ids.forEach(id => { const t = TOOLS_BY_ID[id], R = RUNNERS[t.recipe + ':' + APP.ds]; nav.append(h('button', { 'aria-current': APP.page === id ? 'page' : null, onclick: () => go(id) }, h('span.g', null, t.glyph), t.name, R && R.api ? h('i.dot') : null)); }); });
  nav.append(h('h4', null, 'Evaluate'), h('button', { 'aria-current': APP.page === 'compare' ? 'page' : null, onclick: () => go('compare') }, h('span.g', null, 'FID'), 'Compare & Metrics'), h('button', { 'aria-current': APP.page === 'map' ? 'page' : null, onclick: () => go('map') }, h('span.g', null, 'MAP'), 'Project Map'));
  nav.classList.remove('open');
}
function boot() {
  APP.mode = store.get('mode', 'easy'); APP.ds = store.get('ds', 'digits'); if (!DATASETS[APP.ds]) APP.ds = 'digits';
  const theme = store.get('theme', ''); if (theme) document.documentElement.dataset.theme = theme;
  const app = $('#app');
  const modeSeg = h('div.seg', { role: 'group', 'aria-label': 'Mode' }, [['easy', 'Easy Studio'], ['learn', 'Learn Lab']].map(([k, l]) => h('button', { 'aria-pressed': APP.mode === k, onclick: e => { setMode(k); [...modeSeg.children].forEach(b => b.setAttribute('aria-pressed', b === e.currentTarget)); render(); } }, l)));
  const dsSel = h('select.sel', { 'aria-label': 'Dataset', onchange: e => { APP.ds = e.target.value; store.set('ds', APP.ds); render(); } }, Object.values(DATASETS).map(d => h('option', { value: d.id, selected: d.id === APP.ds }, d.name)));
  const backend = h('span.pill.hide-sm', null, tf.getBackend());
  app.append(h('header.top', null, h('button.btn.burger', { 'aria-label': 'Menu', onclick: () => $('.nav').classList.toggle('open') }, '☰'), h('button.brand', { onclick: () => go('home') }, h('i'), 'GenLab'), h('span.spacer'),
    h('span.hide-sm.muted', null, 'Dataset'), dsSel, modeSeg, backend, h('button.btn.sm', { 'aria-label': 'Toggle theme', onclick: () => { const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'), nx = cur === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = nx; store.set('theme', nx); render(); } }, '◐')),
    h('nav.nav', { 'aria-label': 'Tools' }), h('main.main'));
  const route = () => { const p = location.hash.slice(1) || 'home'; APP.page = p === 'home' || p === 'compare' || p === 'map' || TOOLS_BY_ID[p] ? p : 'home'; render(); };
  window.addEventListener('hashchange', route); route();
  window.__genlab = { APP, getRunner, TOOLS_BY_ID, loadDataset, ZOO, RUNNERS, METRICS, getEvaluator };
}
tf.ready().then(boot);
