// ═══════════════════════════════════════════════════════════════
//  DIFFUSION (DDPM) · learn to remove noise step by step, then generate from pure noise
//  Class-conditional with classifier-free guidance → powers "Text → Image"
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · The forward (noising) process ─────────────────────
const { S, C, K } = data;
const D = S * S * C, T = P.T;
// β_t = how much noise is added at step t. ᾱ_t = fraction of the ORIGINAL image still left at step t.
const betas = Array.from({ length: T }, (_, t) => 1e-3 + (0.2 - 1e-3) * t / (T - 1));
const ab = []; betas.reduce((acc, b) => (ab.push(acc * (1 - b)), acc * (1 - b)), 1);
const alphaBar = tf.tensor1d(ab);
// x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε     (one jump from clean to step t)

// ── STEP 2 · The denoiser network ──────────────────────────────
// Input: noisy image, the step number t, and the class label (K = "no label").
// Output: its guess of the noise ε that was added.
const timeEmb = t => tf.tidy(() => {                     // step number → sin/cos features
  const ang = t.cast('float32').div(T).expandDims(1).mul(tf.range(1, 9).mul(Math.PI));
  return tf.concat([ang.sin(), ang.cos(), t.cast('float32').div(T).expandDims(1)], 1);
});
const oneHot = c => tf.oneHot(c, K + 1).cast('float32');
const xin = tf.input({ shape: [D] }), tin = tf.input({ shape: [17] }), cin = tf.input({ shape: [K + 1] });
const cat = (...l) => tf.layers.concatenate().apply(l);
let hid = tf.layers.dense({ units: P.width, activation: 'swish' }).apply(cat(xin, tin, cin));
for (let i = 0; i < 3; i++) {                            // residual blocks that re-read t and class each time
  const g = tf.layers.dense({ units: P.width, activation: 'swish' }).apply(cat(hid, tin, cin));
  hid = tf.layers.add().apply([hid, g]);
}
const net = tf.model({ inputs: [xin, tin, cin], outputs: tf.layers.dense({ units: D }).apply(hid) });
const params = net.trainableWeights.map(w => w.val);
const opt = tf.train.adam(P.lr);
const flatImgs = data.xTrain.reshape([-1, D]);

// ── STEP 3 · Training: predict the noise ───────────────────────
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => {
    const [xb, y] = lab.batchXY(flatImgs, data.yTrain, 128);
    const x0 = xb.mul(2).sub(1);                         // pixels 0..1 → −1..1
    const t = tf.randomUniform([128], 0, T, 'int32');
    const a = alphaBar.gather(t).expandDims(1);
    const eps = tf.randomNormal([128, D]);
    const xt = a.sqrt().mul(x0).add(tf.sub(1, a).sqrt().mul(eps));
    // classifier-free guidance: hide the label 15% of the time so the net also learns "unconditional"
    const label = tf.where(tf.randomUniform([128]).less(0.15), tf.fill([128], K, 'int32'), y);
    const te = timeEmb(t), ce = oneHot(label);
    return opt.minimize(() => tf.mean(tf.square(tf.sub(eps, net.apply([xt, te, ce], { training: true })))), true, params);
  });
  if (step % 10 === 0) lab.plot('noise-prediction error', (await cost.data())[0]);
  cost.dispose();
  if (step % Math.max(100, Math.floor(P.steps / 8)) === 0 || step === P.steps - 1) {
    const out = sample(16, Array.from({ length: 16 }, (_, i) => i % K), 1.5);
    lab.show(out.images, 'Samples (guidance 1.5)', { cols: 8, key: 'live' }); out.images.dispose(); out.traj.forEach(t => t.dispose());
  }
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 4 · Sampling: start from pure noise, denoise T times ──
// w = guidance strength: ε = ε_uncond + (1+w)·(ε_cond − ε_uncond). Larger w = follows the prompt harder.
function sample(n, classes, w = 1, keepEvery = Math.max(1, Math.floor(T / 7))) {
  let x = tf.randomNormal([n, D]); const traj = [];
  const cls = classes ? tf.tensor1d(Array.from({ length: n }, (_, i) => classes[i % classes.length]), 'int32') : null;
  for (let t = T - 1; t >= 0; t--) {
    if (t % keepEvery === 0 || t === T - 1) traj.push(tf.tidy(() => x.add(1).div(2).clipByValue(0, 1).reshape([-1, S, S, C])));
    const nx = tf.tidy(() => {
      const te = timeEmb(tf.fill([n], t, 'int32')), eU = net.predict([x, te, oneHot(tf.fill([n], K, 'int32'))]);
      const eps = cls ? eU.add(net.predict([x, te, oneHot(cls)]).sub(eU).mul(1 + w)) : eU;
      const a = ab[t], aPrev = t > 0 ? ab[t - 1] : 1, beta = betas[t];
      const x0 = x.sub(eps.mul(Math.sqrt(1 - a))).div(Math.sqrt(a)).clipByValue(-1, 1);       // guess of the clean image
      const mean = x0.mul(Math.sqrt(aPrev) * beta / (1 - a)).add(x.mul(Math.sqrt(1 - beta) * (1 - aPrev) / (1 - a)));
      return t === 0 ? mean : mean.add(tf.randomNormal([n, D]).mul(Math.sqrt(beta * (1 - aPrev) / (1 - a))));
    });
    x.dispose(); x = nx;
  }
  if (cls) cls.dispose();
  const images = tf.tidy(() => x.add(1).div(2).clipByValue(0, 1).reshape([-1, S, S, C])); x.dispose();
  return { images, traj };
}
lab.register({ id: 'diffusion', label: 'Diffusion (DDPM)', sample: async n => { const o = sample(n, Array.from({ length: K }, (_, i) => i), 1); o.traj.forEach(t => t.dispose()); return o.images; } });
return { T, K, sample };
