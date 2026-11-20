// ═══════════════════════════════════════════════════════════════
//  VAE · an autoencoder whose codes form a smooth, samplable space (ELBO)
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · Data ──────────────────────────────────────────────
const { S, C } = data;
const D = S * S * C, Z = P.latent;
const flat = x => x.reshape([-1, D]);
const trainX = flat(data.xTrain), valX = flat(data.xTest);

// ── STEP 2 · Encoder outputs a DISTRIBUTION: mean μ and log-variance log σ² ─
const inp = tf.input({ shape: [D] });
let hid = tf.layers.dense({ units: 256, activation: 'relu' }).apply(inp);
hid = tf.layers.dense({ units: 128, activation: 'relu' }).apply(hid);
const mu = tf.layers.dense({ units: Z, name: 'mu' }).apply(hid);
const logvar = tf.layers.dense({ units: Z, name: 'logvar' }).apply(hid);
const encoder = tf.model({ inputs: inp, outputs: [mu, logvar] });
const decoder = tf.sequential({ layers: [
  tf.layers.dense({ inputShape: [Z], units: 128, activation: 'relu' }),
  tf.layers.dense({ units: 256, activation: 'relu' }),
  tf.layers.dense({ units: D, activation: 'sigmoid' }),
] });

// ── STEP 3 · Reparameterisation trick: z = μ + σ·ε  with ε ~ N(0,1) ─
// Sampling is random, but writing it this way lets gradients flow back through μ and σ.
const sampleZ = (m, lv) => tf.add(m, tf.mul(tf.exp(tf.mul(0.5, lv)), tf.randomNormal(m.shape)));

// ── STEP 4 · The ELBO loss  =  reconstruction error + β · KL ─────
// reconstruction: binary cross-entropy summed over pixels ("did you redraw the image?")
// KL: distance between this image's distribution N(μ,σ²) and the prior N(0,1)
//     ("is your code tidy enough to sample from?")
const recon = (x, p) => tf.neg(tf.sum(tf.add(
  tf.mul(x, tf.log(tf.add(p, 1e-6))), tf.mul(tf.sub(1, x), tf.log(tf.add(tf.sub(1, p), 1e-6)))), 1));
const kl = (m, lv) => tf.mul(-0.5, tf.sum(tf.sub(tf.sub(tf.add(1, lv), tf.square(m)), tf.exp(lv)), 1));
const params = [...encoder.trainableWeights, ...decoder.trainableWeights].map(w => w.val);
const opt = tf.train.adam(P.lr);

// ── STEP 5 · Train ─────────────────────────────────────────────
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => {
    const x = lab.batch(trainX, 128);
    return opt.minimize(() => {
      const [m, lv] = encoder.apply(x, { training: true });
      const xhat = decoder.apply(sampleZ(m, lv), { training: true });
      return tf.mean(tf.add(recon(x, xhat), tf.mul(P.beta, kl(m, lv))));
    }, true, params);
  });
  if (step % 10 === 0) {
    const [r, k] = tf.tidy(() => { const [m, lv] = encoder.predict(valX); return [tf.mean(recon(valX, decoder.predict(m))), tf.mean(kl(m, lv))]; });
    lab.plot('−ELBO (train)', (await cost.data())[0], 'loss');
    lab.plot('reconstruction', (await r.data())[0], 'parts');
    lab.plot('KL', (await k.data())[0], 'parts');
    r.dispose(); k.dispose();
  }
  cost.dispose();
  if (step % 50 === 0) {
    const demo = tf.tidy(() => decoder.predict(tf.randomNormal([16, Z], 0, 1, 'float32', 7)).reshape([-1, S, S, C]));
    lab.show(demo, 'Random samples decoded from z ~ N(0, 1)', { cols: 8, key: 'live' });
    demo.dispose();
  }
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 6 · Export + register for the Compare tab ─────────────
const decode = z => tf.tidy(() => decoder.predict(z).reshape([-1, S, S, C]));
const encode = x => tf.tidy(() => { const [m, lv] = encoder.predict(flat(x)); return { mu: m, logvar: lv }; });
const sample = n => tf.tidy(() => decode(tf.randomNormal([n, Z])));
lab.register({ id: 'vae', label: 'VAE', sample: async n => sample(n) });
return { Z, decode, encode, sample };
