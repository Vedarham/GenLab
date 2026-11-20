// ═══════════════════════════════════════════════════════════════
//  CycleGAN · translate between two image styles WITHOUT paired examples
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · Two UNPAIRED domains ──────────────────────────────
// Domain A = original images (first half of the data).
// Domain B = a *different* half of the data, re-styled. No image in A has a partner in B.
const { S, C } = data;
const restyle = lab.styles[P.style];
const half = data.xTrain.shape[0] / 2;
const A = data.xTrain.slice([0, 0, 0, 0], [half, S, S, C]);
const B = tf.tidy(() => restyle(data.xTrain.slice([half, 0, 0, 0], [half, S, S, C])));
const aTest = data.xTest.slice([0, 0, 0, 0], [8, S, S, C]);
const bTest = tf.tidy(() => restyle(data.xTest.slice([200, 0, 0, 0], [8, S, S, C])));

// ── STEP 2 · Two generators and two discriminators ─────────────
// G: A→B and F: B→A are small convolutional nets (image in, image out).
// D_A and D_B each ask: "is this a real image of my domain?"
const makeGen = n => tf.sequential({ layers: [        // n = a unique name prefix so the optimiser can tell weights apart
  tf.layers.conv2d({ name: n + '1', inputShape: [S, S, C], filters: 24, kernelSize: 3, padding: 'same', activation: 'relu' }),
  tf.layers.conv2d({ name: n + '2', filters: 24, kernelSize: 3, padding: 'same', activation: 'relu' }),
  tf.layers.conv2d({ name: n + '3', filters: C, kernelSize: 3, padding: 'same', activation: 'sigmoid' }),
] });
const makeDisc = n => tf.sequential({ layers: [
  tf.layers.conv2d({ name: n + '1', inputShape: [S, S, C], filters: 24, kernelSize: 3, strides: 2, padding: 'same' }), tf.layers.leakyReLU({ alpha: 0.2 }),
  tf.layers.conv2d({ name: n + '2', filters: 48, kernelSize: 3, strides: 2, padding: 'same' }), tf.layers.leakyReLU({ alpha: 0.2 }),
  tf.layers.flatten(), tf.layers.dense({ name: n + '3', units: 1 }),
] });
const G = makeGen('G'), F = makeGen('F'), DA = makeDisc('DA'), DB = makeDisc('DB');
const vars = (...ms) => ms.flatMap(m => m.trainableWeights.map(w => w.val));
const gVars = vars(G, F), dVars = vars(DA, DB);
const optG = tf.train.adam(P.lr, 0.5), optD = tf.train.adam(P.lr, 0.5);
const app = (m, x) => m.apply(x, { training: true });

// ── STEP 3 · Three losses ──────────────────────────────────────
// adversarial (least-squares): fakes should fool the matching discriminator
// cycle: A → B → A must return to the start  (this is what replaces paired data!)
// identity (optional): feeding a domain-B image to G should change nothing
const mse = (a, b) => tf.mean(tf.square(tf.sub(a, b)));
const l1 = (a, b) => tf.mean(tf.abs(tf.sub(a, b)));

// ── STEP 4 · Train: generators first, then discriminators ──────
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => {
    const a = lab.batch(A, 16), b = lab.batch(B, 16);
    optG.minimize(() => {
      const fb = app(G, a), fa = app(F, b);
      const adv = mse(app(DB, fb), 1).add(mse(app(DA, fa), 1));
      const cyc = l1(a, app(F, fb)).add(l1(b, app(G, fa)));
      let loss = adv.add(cyc.mul(P.lambda));
      if (P.identity) loss = loss.add(l1(b, app(G, b)).add(l1(a, app(F, a))).mul(P.lambda * 0.5));
      return loss;
    }, false, gVars);
    const fb = G.predict(a), fa = F.predict(b);
    return optD.minimize(() => mse(app(DB, b), 1).add(mse(app(DB, fb), 0)).add(mse(app(DA, a), 1)).add(mse(app(DA, fa), 0)).mul(0.5), true, dVars);
  });
  if (step % 10 === 0) {
    // we know the TRUE style function, so we can measure how close G(a) is to it
    const [err, cyc] = tf.tidy(() => [mse(G.predict(aTest), restyle(aTest)), l1(aTest, F.predict(G.predict(aTest)))]);
    lab.plot('style error vs. ground truth', (await err.data())[0]); lab.plot('cycle error', (await cyc.data())[0]);
    err.dispose(); cyc.dispose(); lab.plot('discriminators', (await cost.data())[0], 'adv');
  }
  cost.dispose();
  if (step % 40 === 0) {
    const rows = tf.tidy(() => tf.concat([aTest, G.predict(aTest), F.predict(G.predict(aTest)), restyle(aTest)]));
    lab.show(rows, 'A → B: input · G(input) · back again F(G(input)) · true style', { cols: 8, key: 'ab' }); rows.dispose();
    const rows2 = tf.tidy(() => tf.concat([bTest, F.predict(bTest), G.predict(F.predict(bTest))]));
    lab.show(rows2, 'B → A: styled input · F(input) · back again G(F(input))', { cols: 8, key: 'ba' }); rows2.dispose();
  }
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 5 · Export ────────────────────────────────────────────
return {
  toStyle: x => tf.tidy(() => G.predict(x)), fromStyle: x => tf.tidy(() => F.predict(x)),
  truth: x => tf.tidy(() => restyle(x)), aTest, bTest,
};
