// ═══════════════════════════════════════════════════════════════
//  AUTOENCODER · squeeze an image through a tiny bottleneck, then rebuild it
// ═══════════════════════════════════════════════════════════════
// Already available: tf (TensorFlow.js) · data (the dataset) · P (settings) · lab (plots & galleries)

// ── STEP 1 · Look at the data ──────────────────────────────────
// data.xTrain has shape [N, 16, 16, C]: N images of 16×16 pixels with C colour channels.
// Every pixel is a number from 0 (black) to 1 (white).
const { S, C } = data;
const D = S * S * C;                                     // numbers per image
const flat = x => x.reshape([-1, D]);                    // image grid → one long vector
const trainX = flat(data.xTrain.slice([0, 0, 0, 0], [P.trainSize, S, S, C]));
const valX = flat(data.xTest);                           // never trained on: reveals overfitting
lab.log('training on ' + trainX.shape[0] + ' images with ' + D + ' numbers each');

// ── STEP 2 · Build the encoder and the decoder ─────────────────
// Encoder: image (D numbers) → code z (only P.latent numbers). Decoder: z → image.
const encoder = tf.sequential({ layers: [
  tf.layers.dense({ inputShape: [D], units: 128, activation: 'relu' }),
  tf.layers.dropout({ rate: P.dropout }),                // regularisation: randomly silence neurons
  tf.layers.dense({ units: P.latent }),                  // ← the bottleneck
] });
const decoder = tf.sequential({ layers: [
  tf.layers.dense({ inputShape: [P.latent], units: 128, activation: 'relu' }),
  tf.layers.dense({ units: D, activation: 'sigmoid' }),  // sigmoid keeps pixels in 0..1
] });
const forward = (x, training) => decoder.apply(encoder.apply(x, { training }), { training });

// ── STEP 3 · Define the loss ───────────────────────────────────
// "How different is the rebuilt image from the original?"  → mean squared error.
// L2 weight decay (optional) punishes huge weights, a second kind of regularisation.
const params = [...encoder.trainableWeights, ...decoder.trainableWeights].map(w => w.val);
const mse = (a, b) => tf.mean(tf.square(tf.sub(a, b)));
const l2 = () => tf.addN(params.filter(p => p.rank > 1).map(p => tf.sum(tf.square(p)))).mul(P.l2);
const opt = tf.train.adam(P.lr);

// ── STEP 4 · Train: repeat  forward → loss → gradient → update ─
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => {
    const xb = lab.batch(trainX, 64);                    // a random mini-batch of 64 images
    return opt.minimize(() => mse(xb, forward(xb, true)).add(l2()), true, params);
  });
  if (step % 10 === 0) {
    const val = tf.tidy(() => mse(valX, forward(valX, false)));
    lab.plot('train', (await cost.data())[0]);
    lab.plot('validation', (await val.data())[0]);
    val.dispose();
  }
  cost.dispose();
  if (step % 50 === 0) {                                 // live preview
    const demo = tf.tidy(() => {
      const x = data.xTest.slice([0, 0, 0, 0], [8, S, S, C]);
      return tf.concat([x, forward(flat(x), false).reshape([-1, S, S, C])]);
    });
    lab.show(demo, 'Top: originals · Bottom: rebuilt from ' + P.latent + ' numbers', { cols: 8, key: 'live' });
    demo.dispose();
  }
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 5 · Hand the trained model to the "Try it" panel ─────
const encode = x => tf.tidy(() => encoder.predict(flat(x)));
const decode = z => tf.tidy(() => decoder.predict(z).reshape([-1, S, S, C]));
return { encode, decode, latent: P.latent, reconstruct: x => decode(encode(x)) };
