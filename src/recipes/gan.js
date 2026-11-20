// ═══════════════════════════════════════════════════════════════
//  GAN · a forger (generator) and a detective (discriminator) train each other
//  Switch P.loss to 'wgan' for the Wasserstein version, P.arch to 'dcgan' for convolutions
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · Data ──────────────────────────────────────────────
const { S, C } = data;
const D = S * S * C, Z = P.latent, B = 64;
const wgan = P.loss === 'wgan', conv = P.arch === 'dcgan';
const real = conv ? data.xTrain : data.xTrain.reshape([-1, D]);
const toImg = o => o.reshape([-1, S, S, C]);

// ── STEP 2 · Generator: random noise z → image ─────────────────
const G = conv ? tf.sequential({ layers: [               // DCGAN: upsample with transposed convolutions
  tf.layers.dense({ inputShape: [Z], units: (S / 4) * (S / 4) * 64 }),
  tf.layers.reshape({ targetShape: [S / 4, S / 4, 64] }),
  tf.layers.batchNormalization(), tf.layers.reLU(),
  tf.layers.conv2dTranspose({ filters: 32, kernelSize: 4, strides: 2, padding: 'same' }),
  tf.layers.batchNormalization(), tf.layers.reLU(),
  tf.layers.conv2dTranspose({ filters: C, kernelSize: 4, strides: 2, padding: 'same', activation: 'sigmoid' }),
] }) : tf.sequential({ layers: [
  tf.layers.dense({ inputShape: [Z], units: 256, activation: 'relu' }),
  tf.layers.dense({ units: 512, activation: 'relu' }),
  tf.layers.dense({ units: D, activation: 'sigmoid' }),
] });

// ── STEP 3 · Discriminator (a "critic" in WGAN): image → score ──
const Dn = conv ? tf.sequential({ layers: [
  tf.layers.conv2d({ inputShape: [S, S, C], filters: 32, kernelSize: 4, strides: 2, padding: 'same' }), tf.layers.leakyReLU({ alpha: 0.2 }),
  tf.layers.conv2d({ filters: 64, kernelSize: 4, strides: 2, padding: 'same' }), tf.layers.leakyReLU({ alpha: 0.2 }),
  tf.layers.flatten(), tf.layers.dense({ units: 1 }),
] }) : tf.sequential({ layers: [
  tf.layers.dense({ inputShape: [D], units: 512 }), tf.layers.leakyReLU({ alpha: 0.2 }),
  tf.layers.dense({ units: 256 }), tf.layers.leakyReLU({ alpha: 0.2 }),
  tf.layers.dense({ units: 1 }),
] });
const gVars = G.trainableWeights.map(w => w.val), dVars = Dn.trainableWeights.map(w => w.val);

// ── STEP 4 · The two loss functions ────────────────────────────
// Vanilla GAN: discriminator = "real→1, fake→0";  generator = "make the detective say 1".
// WGAN: the critic just scores (real high, fake low); weights are clipped so the score stays smooth,
//       which makes training far more stable and the loss actually meaningful.
const bce = (label, logits) => tf.losses.sigmoidCrossEntropy(tf.fill(logits.shape, label), logits);
const dLoss = (xr, xf) => wgan ? Dn.apply(xf, { training: true }).mean().sub(Dn.apply(xr, { training: true }).mean())
  : bce(1, Dn.apply(xr, { training: true })).add(bce(0, Dn.apply(xf, { training: true })));
const gLoss = z => wgan ? Dn.apply(G.apply(z, { training: true }), { training: true }).mean().neg()
  : bce(1, Dn.apply(G.apply(z, { training: true }), { training: true }));
const optD = wgan ? tf.train.rmsprop(P.lr) : tf.train.adam(P.lr, 0.5);
const optG = wgan ? tf.train.rmsprop(P.lr) : tf.train.adam(P.lr, 0.5);
const nCritic = wgan ? 3 : 1;
const fixedZ = tf.randomNormal([16, Z]);                 // same noise every time → watch the SAME samples improve

// ── STEP 5 · Training loop: detective first, then forger ───────
for (let step = 0; step < P.steps; step++) {
  let dl, gl;
  for (let k = 0; k < nCritic; k++) {
    dl = tf.tidy(() => {
      const xr = lab.batch(real, B), xf = G.apply(tf.randomNormal([B, Z]), { training: true });
      return optD.minimize(() => dLoss(xr, xf), true, dVars);
    });
    if (wgan) tf.tidy(() => dVars.forEach(v => v.assign(tf.clipByValue(v, -P.clip, P.clip))));   // WGAN weight clipping
    if (k < nCritic - 1) dl.dispose();
  }
  gl = tf.tidy(() => optG.minimize(() => gLoss(tf.randomNormal([B, Z])), true, gVars));
  if (step % 10 === 0) { lab.plot(wgan ? 'critic' : 'discriminator', (await dl.data())[0]); lab.plot('generator', (await gl.data())[0]); }
  dl.dispose(); gl.dispose();
  if (step % 50 === 0 || step === P.steps - 1) {
    const s = tf.tidy(() => toImg(G.predict(fixedZ)));
    lab.show(s, 'Generator samples (same 16 noise vectors)', { cols: 8, key: 'live' });
    if (step % Math.max(50, Math.floor(P.steps / 5)) === 0 || step === P.steps - 1)
      lab.show(s.slice([0, 0, 0, 0], [8, S, S, C]), 'step ' + step, { cols: 8, key: 'tl' + step, group: 'Learning timeline' });
    s.dispose();
  }
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 6 · Export ────────────────────────────────────────────
const decode = z => tf.tidy(() => toImg(G.predict(z)));
const sample = n => tf.tidy(() => decode(tf.randomNormal([n, Z])));
const name = (wgan ? 'WGAN' : 'GAN') + (conv ? ' (conv)' : ' (MLP)');
lab.register({ id: 'gan-' + P.arch + '-' + P.loss, label: name, sample: async n => sample(n) });
return { Z, decode, sample, name };
