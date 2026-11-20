// ═══════════════════════════════════════════════════════════════
//  DENOISING AUTOENCODER · learn to clean images by training on damaged ones
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · Data ──────────────────────────────────────────────
const { S, C } = data;
const D = S * S * C;
const flat = x => x.reshape([-1, D]);
const trainX = flat(data.xTrain), valX = flat(data.xTest);

// ── STEP 2 · A corruption function (this is the whole trick) ───
// We damage a clean image on purpose, then ask the network to give back the CLEAN one.
const corrupt = (x, level) => tf.tidy(() => {
  if (P.noiseType === 'gaussian')                        // random static
    return tf.clipByValue(x.add(tf.randomNormal(x.shape, 0, level)), 0, 1);
  if (P.noiseType === 'mask')                            // random pixels knocked out to black
    return x.mul(tf.randomUniform(x.shape).greater(level).cast('float32'));
  const r = tf.randomUniform(x.shape);                   // salt & pepper
  const salt = r.less(level / 2).cast('float32'), pepper = r.greater(1 - level / 2).cast('float32');
  return x.mul(tf.sub(1, pepper)).mul(tf.sub(1, salt)).add(salt);
});

// ── STEP 3 · Model: same shape as an autoencoder, a bit wider ──
const net = tf.sequential({ layers: [
  tf.layers.dense({ inputShape: [D], units: 256, activation: 'relu' }),
  tf.layers.dropout({ rate: P.dropout }),
  tf.layers.dense({ units: P.latent, activation: 'relu' }),
  tf.layers.dense({ units: 256, activation: 'relu' }),
  tf.layers.dense({ units: D, activation: 'sigmoid' }),
] });
const params = net.trainableWeights.map(w => w.val);
const mse = (a, b) => tf.mean(tf.square(tf.sub(a, b)));
const opt = tf.train.adam(P.lr);

// ── STEP 4 · Train on (damaged input → clean target) ───────────
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => {
    const clean = lab.batch(trainX, 64);
    const noisy = corrupt(clean, P.noise);
    return opt.minimize(() => mse(clean, net.apply(noisy, { training: true })), true, params);
  });
  if (step % 10 === 0) {
    const [v, base] = tf.tidy(() => { const n = corrupt(valX, P.noise); return [mse(valX, net.predict(n)), mse(valX, n)]; });
    lab.plot('after denoising', (await v.data())[0]);
    lab.plot('noisy input (baseline)', (await base.data())[0]);
    v.dispose(); base.dispose();
  }
  cost.dispose();
  if (step % 50 === 0) {
    const demo = tf.tidy(() => {
      const x = data.xTest.slice([0, 0, 0, 0], [8, S, S, C]), n = corrupt(flat(x), P.noise);
      return tf.concat([x, n.reshape([-1, S, S, C]), net.predict(n).reshape([-1, S, S, C])]);
    });
    lab.show(demo, 'Rows: clean · damaged · cleaned by the network', { cols: 8, key: 'live' });
    demo.dispose();
  }
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 5 · Export ────────────────────────────────────────────
const shape4 = t => t.reshape([-1, S, S, C]);
return {
  corrupt: (x, level) => tf.tidy(() => shape4(corrupt(flat(x), level))),
  denoise: x => tf.tidy(() => shape4(net.predict(flat(x)))),
};
