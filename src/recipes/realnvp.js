// ═══════════════════════════════════════════════════════════════
//  RealNVP · an invertible network: images ⇄ Gaussian noise, with EXACT likelihood
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · Data ──────────────────────────────────────────────
const { S, C } = data;
const D = S * S * C, L = P.layers;
const prep = x => x.mul(2).sub(1);                       // 0..1 → −1..1
const trainX = prep(data.xTrain.reshape([-1, D])), testX = prep(data.xTest.reshape([-1, D]));

// ── STEP 2 · Masks: each layer transforms half the pixels, using the other half ─
const makeMask = kind => {
  const m = new Float32Array(D);
  for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) for (let k = 0; k < C; k++) {
    const i = (r * S + c) * C + k;
    m[i] = kind === 0 ? (r + c) % 2 : kind === 1 ? 1 - (r + c) % 2 : kind === 2 ? (c < S / 2 ? 1 : 0) : (c < S / 2 ? 0 : 1);
  }
  return tf.tensor1d(m);
};
const masks = Array.from({ length: L }, (_, l) => makeMask(l % 4));   // checkerboard, inverse, left half, right half…

// ── STEP 3 · Affine coupling layers ────────────────────────────
// Frozen half passes through. The other half is scaled and shifted by amounts (s, t)
// computed from the frozen half:   y = x · exp(s) + t   →   invertible by:   x = (y − t) · exp(−s)
const nets = masks.map(() => tf.sequential({ layers: [
  tf.layers.dense({ inputShape: [D], units: P.hidden, activation: 'relu' }),
  tf.layers.dense({ units: P.hidden, activation: 'relu' }),
  tf.layers.dense({ units: 2 * D, kernelInitializer: 'zeros' }),   // start as the identity map
] }));
const st = (l, frozen) => {
  const o = nets[l].apply(frozen.mul(masks[l]), { training: true }), free = tf.sub(1, masks[l]);
  return [tf.tanh(o.slice([0, 0], [-1, D])).mul(free), o.slice([0, D], [-1, D]).mul(free)];
};
function forward(x) {                                   // image → noise, also returns log|det J|
  let z = x, logdet = tf.zeros([x.shape[0]]);
  for (let l = 0; l < L; l++) { const [s, t] = st(l, z); z = z.mul(s.exp()).add(t); logdet = logdet.add(s.sum(1)); }
  return { z, logdet };
}
function inverse(z) {                                   // noise → image
  let x = z;
  for (let l = L - 1; l >= 0; l--) { const [s, t] = st(l, x); x = x.sub(t).mul(s.neg().exp()); }
  return x;
}

// ── STEP 4 · Loss = negative log-likelihood (change-of-variables formula) ─
// log p(x) = log N(z; 0, I) + log|det ∂z/∂x|      We train by maximising it.
const nll = x => { const { z, logdet } = forward(x); return z.square().sum(1).mul(0.5).sub(logdet).div(D).add(0.5 * Math.log(2 * Math.PI)); };
const params = nets.flatMap(n => n.trainableWeights.map(w => w.val));
const opt = tf.train.adam(P.lr);
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => {
    const x = lab.batch(trainX, 128).add(tf.randomNormal([128, D], 0, 0.05));     // tiny dequantisation noise
    return opt.minimize(() => nll(x).mean(), true, params);
  });
  if (step % 10 === 0) {
    const v = tf.tidy(() => nll(testX.add(tf.randomNormal(testX.shape, 0, 0.05))).mean());
    lab.plot('train', (await cost.data())[0]); lab.plot('test', (await v.data())[0]); v.dispose();
  }
  cost.dispose();
  if (step % 100 === 0 || step === P.steps - 1) {
    const s = tf.tidy(() => sample(16, 0.7)); lab.show(s, 'Samples (temperature 0.7)', { cols: 8, key: 'live' }); s.dispose();
  }
  if (!(await lab.tick(step, P.steps))) break;
}
// ── STEP 5 · Sampling = run the network backwards ──────────────
function sample(n, temp = 0.7) {
  return tf.tidy(() => inverse(tf.randomNormal([n, D], 0, temp)).add(1).div(2).clipByValue(0, 1).reshape([-1, S, S, C]));
}
lab.register({ id: 'realnvp', label: 'RealNVP', sample: async n => sample(n, 0.8) });
const toZ = x => tf.tidy(() => forward(prep(x.reshape([-1, D]))).z);
const fromZ = z => tf.tidy(() => inverse(z).add(1).div(2).clipByValue(0, 1).reshape([-1, S, S, C]));
return { sample, toZ, fromZ, bitsPerDim: () => tf.tidy(() => nll(testX).mean().div(Math.LN2).dataSync()[0]) };
