'use strict';
/* ═══════════════ GenLab core: datasets, metrics, runner ═══════════════ */
const $ = (s, r = document) => r.querySelector(s);
function h(tag, props, ...kids) {
  const m = /^([a-z0-9]+)((?:[.#][\w-]+)*)$/i.exec(tag) || [0, 'div', ''];
  const e = document.createElement(m[1]);
  (m[2].match(/[.#][\w-]+/g) || []).forEach(t => t[0] === '.' ? e.classList.add(t.slice(1)) : e.id = t.slice(1));
  if (props) for (const k in props) {
    const v = props[k];
    if (v == null || v === false) continue;
    if (k.startsWith('on')) e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'class') e.className += ' ' + v;
    else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
    else if (k === 'html') e.innerHTML = v;
    else if (k in e && k !== 'list') { try { e[k] = v; } catch (_) { e.setAttribute(k, v); } }
    else e.setAttribute(k, v === true ? '' : v);
  }
  const add = k => { if (k == null || k === false) return; if (Array.isArray(k)) k.forEach(add); else e.append(k.nodeType ? k : document.createTextNode(String(k))); };
  kids.forEach(add);
  return e;
}
function rngFrom(seed) { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const pick = (r, arr) => arr[Math.floor(r() * arr.length)];
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ───────── datasets (procedurally drawn on a canvas, 16×16) ───────── */
const NUM = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const SHAPES = ['circle', 'square', 'triangle', 'cross', 'ring', 'diamond'];
const HAIR = ['black', 'brown', 'blond', 'red'], MOOD = ['smiling', 'frowning'];
const FACE_CLASSES = []; HAIR.forEach(hh => MOOD.forEach(m => FACE_CLASSES.push(m + ' ' + hh + ' hair')));
const DATASETS = {
  digits: { id: 'digits', name: 'Digits', S: 16, C: 1, K: 10, classes: NUM.map((n, i) => String(i)), blurb: 'Ten digit classes drawn in many fonts, tilts and thicknesses. Grayscale 16×16.' },
  shapes: { id: 'shapes', name: 'Shapes', S: 16, C: 1, K: 6, classes: SHAPES, blurb: 'Six white shapes at random sizes, positions and rotations. Grayscale 16×16.' },
  faces: { id: 'faces', name: 'Cartoon faces', S: 16, C: 3, K: 8, classes: FACE_CLASSES, blurb: 'Colour cartoon portraits: 4 hair colours × smiling or frowning, with random skin tones and backgrounds. RGB 16×16.' },
};
const _cv = document.createElement('canvas'); _cv.width = _cv.height = 64;
const _cx = _cv.getContext('2d', { willReadFrequently: true });
function downsample(S, C, out, off) {
  const f = 64 / S, im = _cx.getImageData(0, 0, 64, 64).data;
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    let r = 0, g = 0, b = 0;
    for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) { const p = ((y * f + j) * 64 + x * f + i) * 4; r += im[p]; g += im[p + 1]; b += im[p + 2]; }
    const n = f * f * 255, o = off + (y * S + x) * C;
    if (C === 1) out[o] = (0.299 * r + 0.587 * g + 0.114 * b) / n; else { out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; }
  }
}
const FONTS = ['Arial', 'Georgia', 'Courier New', 'Verdana', 'Times New Roman', 'Trebuchet MS', 'sans-serif', 'serif', 'monospace'];
function drawDigit(c, d, r) {
  c.fillStyle = '#000'; c.fillRect(0, 0, 64, 64); c.save();
  c.translate(32 + (r() - .5) * 8, 33 + (r() - .5) * 8); c.rotate((r() - .5) * .5); c.transform(1, 0, (r() - .5) * .4, 1, 0, 0);
  const size = 52 + r() * 14; c.font = (r() < .5 ? 'bold ' : '') + size + 'px ' + pick(r, FONTS);
  c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = '#fff'; c.fillText(String(d), 0, size * .04); c.restore();
}
function drawShape(c, k, r) {
  c.fillStyle = '#000'; c.fillRect(0, 0, 64, 64); c.save();
  c.translate(32 + (r() - .5) * 12, 32 + (r() - .5) * 12); c.rotate(r() * Math.PI); const R = 14 + r() * 12;
  c.fillStyle = c.strokeStyle = '#fff'; c.beginPath();
  if (k === 0) c.arc(0, 0, R, 0, 7);
  else if (k === 1) c.rect(-R * .8, -R * .8, R * 1.6, R * 1.6);
  else if (k === 2) { for (let i = 0; i < 3; i++) { const a = i * 2.094 - 1.57; c.lineTo(Math.cos(a) * R * 1.1, Math.sin(a) * R * 1.1); } c.closePath(); }
  else if (k === 3) { const t = R * .3; c.rect(-R, -t, 2 * R, 2 * t); c.rect(-t, -R, 2 * t, 2 * R); }
  else if (k === 4) { c.lineWidth = 7; c.arc(0, 0, R * .8, 0, 7); c.stroke(); c.restore(); return; }
  else { for (let i = 0; i < 4; i++) { const a = i * 1.5708; c.lineTo(Math.cos(a) * R * 1.2, Math.sin(a) * R * 1.2); } c.closePath(); }
  c.fill(); c.restore();
}
const SKIN = ['#f6d3b3', '#e8b98d', '#c68b59', '#8d5a3b', '#f1c9a5'];
const BGS = [['blue', '#a9c7f0'], ['green', '#b5e0b0'], ['pink', '#f2b8d0'], ['gray', '#cfd3d9'], ['yellow', '#f3e6a0']];
const HAIRC = ['#1b1b1f', '#6b3f1f', '#e6c25a', '#b8431f'];
const jit = (hex, r, a = 14) => { const n = parseInt(hex.slice(1), 16); return 'rgb(' + [n >> 16, n >> 8 & 255, n & 255].map(v => Math.max(0, Math.min(255, v + (r() - .5) * a * 2))).join(',') + ')'; };
function drawFace(c, cls, r) {
  const hair = cls >> 1, smile = (cls & 1) === 0, bg = Math.floor(r() * BGS.length);
  c.fillStyle = BGS[bg][1]; c.fillRect(0, 0, 64, 64);
  const fx = 32 + (r() - .5) * 5, fy = 37 + (r() - .5) * 4, rx = 16 + r() * 3, ry = 19 + r() * 3, hc = jit(HAIRC[hair], r);
  if (r() < .5) { c.fillStyle = hc; c.beginPath(); c.ellipse(fx, fy - 2, rx + 5, ry + 8, 0, 0, 7); c.fill(); }
  c.fillStyle = jit(pick(r, SKIN), r, 6); c.beginPath(); c.ellipse(fx, fy, rx, ry, 0, 0, 7); c.fill();
  c.fillStyle = hc; c.beginPath(); c.ellipse(fx, fy - 8, rx + 2, ry - 5, 0, Math.PI, 2 * Math.PI); c.fill();
  c.fillStyle = '#1d1d22'; [-1, 1].forEach(s => { c.beginPath(); c.arc(fx + s * 7, fy - 2 + (r() - .5), 2.8, 0, 7); c.fill(); });
  c.strokeStyle = '#7a1f2b'; c.lineWidth = 3.2; c.lineCap = 'round'; c.beginPath();
  if (smile) c.arc(fx, fy + 5, 7.5, .15 * Math.PI, .85 * Math.PI); else c.arc(fx, fy + 16, 7.5, 1.15 * Math.PI, 1.85 * Math.PI);
  c.stroke(); return BGS[bg][0];
}
const _dsCache = {};
async function loadDataset(id) {
  if (_dsCache[id]) return _dsCache[id];
  const spec = DATASETS[id], { S, C, K } = spec, nTrain = 2000, nTest = 400, N = nTrain + nTest, D = S * S * C;
  const r = rngFrom(4242 + id.length * 31), X = new Float32Array(N * D), Y = new Int32Array(N), meta = [];
  for (let i = 0; i < N; i++) {
    const k = i % K; Y[i] = k; let m = {};
    if (id === 'digits') drawDigit(_cx, k, r); else if (id === 'shapes') drawShape(_cx, k, r); else m.bg = drawFace(_cx, k, r);
    downsample(S, C, X, i * D); meta.push(m);
    if (i % 300 === 299) await sleep(0);
  }
  const ink = new Float32Array(N); for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < D; j++) s += X[i * D + j]; ink[i] = s; }
  const med = Array.from(ink).sort((a, b) => a - b)[N >> 1], cr = rngFrom(99), caps = [];
  for (let i = 0; i < N; i++) {
    const k = Y[i], hi = ink[i] > med; let t;
    if (id === 'digits') { const w = hi ? 'bold' : 'thin', n = NUM[k]; t = pick(cr, ['a ' + w + ' ' + n, 'the digit ' + n + ' in ' + w + ' strokes', 'a ' + w + ' number ' + n]); }
    else if (id === 'shapes') { const w = hi ? 'big' : 'small', n = SHAPES[k]; t = pick(cr, ['a ' + w + ' ' + n, 'the ' + n + ' looks ' + w, 'a ' + w + ' white ' + n + ' on black']); }
    else { const hair = HAIR[k >> 1], mood = MOOD[k & 1], bg = meta[i].bg; t = pick(cr, ['a ' + mood + ' ' + hair + ' haired person on a ' + bg + ' background', hair + ' hair ' + mood + ' ' + bg + ' background', 'a ' + bg + ' picture of a ' + mood + ' person with ' + hair + ' hair']); }
    caps.push(t);
  }
  const words = new Set(); caps.forEach(t => t.split(' ').forEach(w => words.add(w)));
  const vocab = ['<pad>', '<bos>', '<eos>', ...[...words].sort()], maxLen = Math.max(...caps.map(t => t.split(' ').length)) + 2;
  const tok = {
    vocab, maxLen, PAD: 0, BOS: 1, EOS: 2,
    encode(t) { const ids = [1, ...t.split(' ').map(w => vocab.indexOf(w)), 2]; while (ids.length < maxLen) ids.push(0); return ids; },
    decode(ids) { const o = []; for (const i of ids) { if (i === 2) break; if (i > 2) o.push(vocab[i]); } return o.join(' '); },
  };
  const capIds = new Int32Array(N * maxLen); caps.forEach((t, i) => capIds.set(tok.encode(t), i * maxLen));
  const T = (a, n, o, s) => tf.tensor(a.subarray(o * s, (o + n) * s), [n, ...(s === D ? [S, S, C] : [maxLen])], a instanceof Int32Array ? 'int32' : 'float32');
  const ds = {
    ...spec, D, nTrain, nTest, tok, meta,
    xTrain: T(X, nTrain, 0, D), xTest: T(X, nTest, nTrain, D),
    yTrain: tf.tensor1d(Y.subarray(0, nTrain), 'int32'), yTest: tf.tensor1d(Y.subarray(nTrain), 'int32'),
    yTrainArr: Y.subarray(0, nTrain), yTestArr: Y.subarray(nTrain),
    capTrain: T(capIds, nTrain, 0, maxLen), capTest: T(capIds, nTest, nTrain, maxLen),
    captions: { train: caps.slice(0, nTrain), test: caps.slice(nTrain) },
  };
  return _dsCache[id] = ds;
}
/* text prompt → class ids (keyword encoder; stands in for a real text transformer) */
const PROMPT_WORDS = {
  digits: () => { const m = {}; NUM.forEach((n, i) => { m[n] = [i]; m[String(i)] = [i]; }); m.nine = [9]; return { classes: m }; },
  shapes: () => ({ classes: { circle: [0], round: [0], ball: [0], dot: [0], square: [1], box: [1], triangle: [2], cross: [3], plus: [3], ring: [4], donut: [4], hoop: [4], diamond: [5], rhombus: [5] } }),
  faces: () => ({ hair: { black: 0, brown: 1, brunette: 1, blond: 2, blonde: 2, yellow: 2, red: 3, ginger: 3 }, mood: { smile: 0, smiling: 0, happy: 0, smiley: 0, frown: 1, frowning: 1, sad: 1, grumpy: 1, angry: 1 } }),
};
function textToClasses(dsId, prompt) {
  const ws = prompt.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).flatMap(w => [w, w.replace(/s$/, '')]), spec = PROMPT_WORDS[dsId]();
  if (dsId !== 'faces') { const out = []; ws.forEach(w => { if (spec.classes[w]) spec.classes[w].forEach(c => { if (!out.includes(c)) out.push(c); }); }); return out.length ? out : null; }
  let hs = [], ms = []; ws.forEach(w => { if (w in spec.hair && !hs.includes(spec.hair[w])) hs.push(spec.hair[w]); if (w in spec.mood && !ms.includes(spec.mood[w])) ms.push(spec.mood[w]); });
  if (!hs.length && !ms.length) return null; if (!hs.length) hs = [0, 1, 2, 3]; if (!ms.length) ms = [0, 1];
  const out = []; hs.forEach(a => ms.forEach(b => out.push(a * 2 + b))); return out;
}
const PROMPT_HINTS = { digits: ['a seven', 'three', 'the number 8', '0 and 5'], shapes: ['a circle', 'a ring', 'square and cross', 'a diamond'], faces: ['a smiling blond person', 'sad red hair', 'frowning brown hair', 'a happy face with black hair'] };

/* ───────── style transforms for CycleGAN domain B ───────── */
const STYLES = {
  negative: { label: 'Negative (invert colours)', fn: x => tf.sub(1, x) },
  edges: { label: 'Sketch (outlines)', fn: x => tf.tidy(() => {
    const g = x.mean(3, true), kx = tf.tensor4d([-1, 0, 1, -2, 0, 2, -1, 0, 1], [3, 3, 1, 1]), ky = tf.tensor4d([-1, -2, -1, 0, 0, 0, 1, 2, 1], [3, 3, 1, 1]);
    const m = tf.sqrt(tf.add(tf.square(tf.conv2d(g, kx, 1, 'same')), tf.square(tf.conv2d(g, ky, 1, 'same'))).add(1e-6));
    const e = tf.clipByValue(tf.sub(1, m.mul(1.2)), 0, 1); return x.shape[3] === 3 ? tf.tile(e, [1, 1, 1, 3]) : e; }) },
  posterize: { label: 'Poster (3 flat levels)', fn: x => tf.round(x.mul(2)).div(2) },
  colorswap: { label: 'Colour swap (R→G→B, faces only)', fn: x => x.shape[3] === 3 ? tf.gather(x, [2, 0, 1], 3) : x },
};

/* ───────── metrics (this exact source is shown in the Learn tab and executed) ───────── */
const METRICS_SRC = `// Mini-FID, Inception Score, diversity and mode-collapse statistics.
// Same maths as the published metrics. The only change: image features come from a small
// classifier trained on THIS dataset instead of Inception-v3 (which needs 299×299 photos).

function meanCov(F) {                         // F = n feature vectors, each of length d
  const n = F.length, d = F[0].length, mu = new Float64Array(d), C = new Float64Array(d * d);
  for (const f of F) for (let i = 0; i < d; i++) mu[i] += f[i] / n;
  for (const f of F) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i * d + j] += (f[i] - mu[i]) * (f[j] - mu[j]) / (n - 1);
  return { mu, C, d };
}
function eigSym(A, d) {                       // Jacobi eigenvalue method for symmetric matrices
  const a = Float64Array.from(A), V = new Float64Array(d * d);
  for (let i = 0; i < d; i++) V[i * d + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0; for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) off += a[i * d + j] ** 2;
    if (off < 1e-18) break;
    for (let p = 0; p < d - 1; p++) for (let q = p + 1; q < d; q++) {
      const apq = a[p * d + q]; if (Math.abs(apq) < 1e-14) continue;
      const th = (a[q * d + q] - a[p * d + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < d; k++) { const x = a[k * d + p], y = a[k * d + q]; a[k * d + p] = c * x - s * y; a[k * d + q] = s * x + c * y; }
      for (let k = 0; k < d; k++) { const x = a[p * d + k], y = a[q * d + k]; a[p * d + k] = c * x - s * y; a[q * d + k] = s * x + c * y; }
      for (let k = 0; k < d; k++) { const x = V[k * d + p], y = V[k * d + q]; V[k * d + p] = c * x - s * y; V[k * d + q] = s * x + c * y; }
    }
  }
  return { vals: Array.from({ length: d }, (_, i) => a[i * d + i]), V };
}
function matmul(A, B, d) { const R = new Float64Array(d * d); for (let i = 0; i < d; i++) for (let k = 0; k < d; k++) { const x = A[i * d + k]; for (let j = 0; j < d; j++) R[i * d + j] += x * B[k * d + j]; } return R; }
function fid(r, g) {                          // Fréchet distance between two Gaussians
  const d = r.d; let dm = 0, tr = 0;
  for (let i = 0; i < d; i++) { dm += (r.mu[i] - g.mu[i]) ** 2; tr += r.C[i * d + i] + g.C[i * d + i]; }
  const { vals, V } = eigSym(r.C, d), S = new Float64Array(d * d);       // S = sqrt(C_real)
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) { let s = 0; for (let k = 0; k < d; k++) s += V[i * d + k] * Math.sqrt(Math.max(vals[k], 0)) * V[j * d + k]; S[i * d + j] = s; }
  const M = matmul(matmul(S, g.C, d), S, d); for (let i = 0; i < d; i++) for (let j = i; j < d; j++) { const m = (M[i * d + j] + M[j * d + i]) / 2; M[i * d + j] = M[j * d + i] = m; }
  const tsq = eigSym(M, d).vals.reduce((s, v) => s + Math.sqrt(Math.max(v, 0)), 0);   // Tr (C1 C2)^(1/2)
  return dm + tr - 2 * tsq;
}
function inceptionScore(P) {                  // P = n probability vectors from the classifier
  const n = P.length, K = P[0].length, py = new Float64Array(K);
  for (const p of P) for (let k = 0; k < K; k++) py[k] += p[k] / n;
  let kl = 0; for (const p of P) for (let k = 0; k < K; k++) if (p[k] > 1e-9) kl += p[k] * (Math.log(p[k]) - Math.log(py[k] + 1e-12));
  return Math.exp(kl / n);                    // high = confident AND varied
}
function classStats(P) {                      // mode-collapse detector
  const K = P[0].length, cnt = new Array(K).fill(0); for (const p of P) cnt[p.indexOf(Math.max(...p))]++;
  const share = cnt.map(c => c / P.length); let H = 0; share.forEach(s => { if (s > 0) H -= s * Math.log(s); });
  return { share, entropy: H / Math.log(K), covered: share.filter(s => s >= 0.4 / K).length, K };
}
function diversity(X, pairs = 300) {          // average pixel distance between random pairs
  let s = 0; for (let t = 0; t < pairs; t++) { const a = X[Math.floor(Math.random() * X.length)], b = X[Math.floor(Math.random() * X.length)]; let d = 0; for (let i = 0; i < a.length; i++) d += (a[i] - b[i]) ** 2; s += Math.sqrt(d); }
  return s / pairs;
}
return { meanCov, eigSym, fid, inceptionScore, classStats, diversity };`;
const METRICS = new Function(METRICS_SRC)();

/* evaluator: small classifier whose hidden layer acts like Inception features */
const _evalCache = {};
async function getEvaluator(ds, log) {
  if (_evalCache[ds.id]) return _evalCache[ds.id];
  const D = ds.D, inp = tf.input({ shape: [D] });
  const h1 = tf.layers.dense({ units: 128, activation: 'relu' }).apply(inp), dr = tf.layers.dropout({ rate: .2 }).apply(h1);
  const feat = tf.layers.dense({ units: 32, activation: 'relu', name: 'feat' }).apply(dr), out = tf.layers.dense({ units: ds.K, activation: 'softmax' }).apply(feat);
  const train = tf.model({ inputs: inp, outputs: out }), fm = tf.model({ inputs: inp, outputs: [feat, out] });
  train.compile({ optimizer: tf.train.adam(0.003), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  log && log('Training the evaluator classifier (used for FID / IS)…');
  const X = ds.xTrain.reshape([-1, D]), Y = tf.oneHot(ds.yTrain, ds.K);
  await train.fit(X, Y, { epochs: 8, batchSize: 64, shuffle: true });
  const ev = train.evaluate(ds.xTest.reshape([-1, D]), tf.oneHot(ds.yTest, ds.K)), acc = (await ev[1].data())[0];
  const api = {
    acc, ds,
    async features(x) {   // x [n,S,S,C] -> {F:[n][32], P:[n][K], flat:[n][D]}
      const [f, p] = tf.tidy(() => fm.predict(x.reshape([-1, D])));
      const F = await f.array(), P = await p.array(), flat = await x.reshape([-1, D]).array(); f.dispose(); p.dispose(); return { F, P, flat };
    },
  };
  return _evalCache[ds.id] = api;
}
function fitToDataset(x, ds) {   // resize / channel-match any generator output to the dataset's shape
  return tf.tidy(() => {
    let y = x; if (y.shape[1] !== ds.S) y = tf.image.resizeNearestNeighbor(y, [ds.S, ds.S]);
    if (y.shape[3] !== ds.C) y = ds.C === 1 ? y.mean(3, true) : tf.tile(y, [1, 1, 1, 3]);
    return tf.clipByValue(y, 0, 1);
  });
}

/* ───────── drawing helpers ───────── */
function gridCanvas(t, cols, scale = 4, gap = 2) {
  const [n, hh, ww, c] = t.shape, d = t.dataSync(), rows = Math.ceil(n / cols), cell = ww * scale;
  const cv = document.createElement('canvas'); cv.width = cols * (cell + gap) + gap; cv.height = rows * (hh * scale + gap) + gap;
  const g = cv.getContext('2d'), tmp = document.createElement('canvas'); tmp.width = ww; tmp.height = hh;
  const tg = tmp.getContext('2d'), im = tg.createImageData(ww, hh); g.imageSmoothingEnabled = false;
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < ww * hh; p++) for (let k = 0; k < 3; k++) im.data[p * 4 + k] = 255 * Math.min(1, Math.max(0, d[(i * ww * hh + p) * c + (c === 1 ? 0 : k)]));
    for (let p = 0; p < ww * hh; p++) im.data[p * 4 + 3] = 255;
    tg.putImageData(im, 0, 0);
    g.drawImage(tmp, gap + (i % cols) * (cell + gap), gap + Math.floor(i / cols) * (hh * scale + gap), cell, hh * scale);
  }
  cv.style.maxWidth = '100%'; cv.className = 'pix'; return cv;
}
const PALETTE = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)'];
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function drawChart(cv, chart) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 400, H = cv.clientHeight || 180;
  cv.width = W * dpr; cv.height = H * dpr; const g = cv.getContext('2d'); g.scale(dpr, dpr);
  const names = Object.keys(chart.series); const ink2 = cssVar('--ink-2'), grid = cssVar('--line');
  const padL = 44, padR = 12, padT = 24, padB = 20; let lo = Infinity, hi = -Infinity, nmax = 1;
  names.forEach(n => chart.series[n].forEach(v => { if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } nmax = Math.max(nmax, chart.series[n].length); }));
  if (!isFinite(lo)) { g.fillStyle = ink2; g.font = '12px ui-monospace,monospace'; g.fillText('waiting for data…', padL, H / 2); return; }
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; } const pad = (hi - lo) * .06; lo -= pad; hi += pad;
  const X = i => padL + (W - padL - padR) * (nmax <= 1 ? 0 : i / (nmax - 1)), Y = v => padT + (H - padT - padB) * (1 - (v - lo) / (hi - lo));
  g.font = '11px ui-monospace,Menlo,monospace'; g.textBaseline = 'middle'; g.lineWidth = 1;
  for (let i = 0; i <= 3; i++) { const v = lo + (hi - lo) * i / 3, y = Y(v); g.strokeStyle = grid; g.beginPath(); g.moveTo(padL, y); g.lineTo(W - padR, y); g.stroke(); g.fillStyle = ink2; g.textAlign = 'right'; g.fillText(Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3), padL - 6, y); }
  g.textAlign = 'center'; g.fillText('0', padL, H - 8); g.fillText(String(nmax - 1) + (chart.xlabel ? ' ' + chart.xlabel : ''), W - padR - 20, H - 8);
  let lx = padL;
  names.forEach((n, si) => {
    const col = cssVar('--s' + (si % 5 + 1)), ys = chart.series[n], a = Math.min(.5, 8 / Math.max(8, ys.length / 4)); let sm = null;
    g.strokeStyle = col; g.globalAlpha = .25; g.lineWidth = 1; g.beginPath(); ys.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); g.globalAlpha = 1;
    g.lineWidth = 2; g.lineJoin = 'round'; g.beginPath(); ys.forEach((v, i) => { sm = sm == null ? v : sm + (v - sm) * Math.max(.08, a * .6); i ? g.lineTo(X(i), Y(sm)) : g.moveTo(X(i), Y(sm)); }); g.stroke();
    if (ys.length) { g.fillStyle = col; g.beginPath(); g.arc(X(ys.length - 1), Y(sm), 3.5, 0, 7); g.fill(); }
    g.fillStyle = col; g.fillRect(lx, 8, 10, 3); g.fillStyle = cssVar('--ink'); g.textAlign = 'left'; g.fillText(n, lx + 15, 10); lx += 22 + g.measureText(n).width + 12;
  });
}

/* ───────── Runner: executes a recipe (the visible code) and stores its outputs ───────── */
const AsyncFunction = Object.getPrototypeOf(async function () { }).constructor;
const ZOO = {};   // `${ds}:${id}` -> {label, sample, ds, size}
const RUNNERS = {};
class Runner {
  constructor(tool, dsId) {
    this.tool = tool; this.ds = dsId; this.code = RECIPES[tool.recipe]; this.params = {}; tool.params.forEach(p => this.params[p.id] = p.def);
    Object.assign(this.params, tool.defaults && tool.defaults[dsId] || {});
    this.charts = {}; this.gallery = []; this.logs = []; this.status = 'idle'; this.progress = 0; this.api = null; this.subs = new Set(); this.stop = false; this.t0 = 0;
  }
  on(f) { this.subs.add(f); return () => this.subs.delete(f); }
  emit() { this.subs.forEach(f => { try { f(this); } catch (e) { console.error(e); } }); }
  log(m, kind) { this.logs.push({ m: String(m), kind: kind || '' }); if (this.logs.length > 200) this.logs.shift(); this.emit(); }
  makeLab(data) {
    const R = this; let last = 0;
    return {
      tf, data, styles: Object.fromEntries(Object.entries(STYLES).map(([k, v]) => [k, v.fn])),
      log: (...a) => R.log(a.join(' ')),
      plot(series, y, chart = 'loss') { const c = R.charts[chart] || (R.charts[chart] = { series: {}, xlabel: 'checks' }); (c.series[series] || (c.series[series] = [])).push(y); },
      show(t, title, o = {}) {
        const cv = gridCanvas(t, o.cols || t.shape[0], o.scale || 5), key = o.key || title, e = { key, title, group: o.group || 'Results', canvas: cv, note: o.note };
        const i = R.gallery.findIndex(g => g.key === key); if (i >= 0) R.gallery[i] = e; else R.gallery.push(e);
      },
      async tick(i, total) { R.progress = (i + 1) / total; const n = performance.now(); if (n - last > 70) { last = n; R.emit(); await tf.nextFrame(); } return !R.stop; },
      batch: (x, n) => tf.tidy(() => tf.gather(x, tf.randomUniform([n], 0, x.shape[0], 'int32'))),
      batchXY: (x, y, n) => tf.tidy(() => { const idx = tf.randomUniform([n], 0, x.shape[0], 'int32'); return [tf.gather(x, idx), tf.gather(y, idx)]; }),
      register(o) { ZOO[R.ds + ':' + o.id] = { ...o, ds: R.ds }; },
    };
  }
  reset() { this.charts = {}; this.gallery = []; this.logs = []; this.progress = 0; this.api = null; }
  async run() {
    if (this.status === 'running') return;
    this.reset(); this.status = 'running'; this.stop = false; this.t0 = performance.now(); this.emit();
    try {
      const data = await loadDataset(this.ds), lab = this.makeLab(data);
      const fn = new AsyncFunction('tf', 'data', 'P', 'lab', this.code);
      this.api = await fn(tf, data, { ...this.params }, lab) || null;
      this.status = this.stop ? 'stopped' : 'done'; this.progress = 1;
      this.log((this.stop ? 'Stopped' : 'Finished') + ' in ' + ((performance.now() - this.t0) / 1000).toFixed(1) + 's', 'ok');
    } catch (e) {
      this.status = 'error'; const m = /<anonymous>:(\d+):(\d+)/.exec(e.stack || ''); this.log((m ? 'line ' + (m[1] - 2) + ': ' : '') + e.message, 'err'); console.error(e);
    }
    this.emit();
  }
}
function getRunner(tool, dsId) { const k = tool.recipe + ':' + dsId; return RUNNERS[k] || (RUNNERS[k] = new Runner(TOOLS_BY_RECIPE[tool.recipe], dsId)); }
