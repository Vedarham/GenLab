// ═══════════════════════════════════════════════════════════════
//  PixelRNN · paint an image one pixel at a time, each pixel predicted from all previous ones
//  (8×8 black-and-white version, reading pixels left→right, top→bottom with a GRU)
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · Shrink to 8×8 and binarise: each pixel is now 0 or 1 ─
const R = 8, L = R * R;
const toBits = x => tf.tidy(() => {
  const g = tf.image.resizeBilinear(x, [R, R]).mean(3).reshape([-1, L]);
  return g.greater(g.mean(1, true).mul(P.sens)).cast('float32');      // brighter than the image's average → 1
});
const bits = toBits(data.xTrain), testBits = toBits(data.xTest);

// ── STEP 2 · Chain rule: p(image) = p(x1)·p(x2|x1)·p(x3|x1,x2)·… ─
// The GRU reads the pixels so far (plus the row/column of the pixel to predict)
// and outputs the probability that the NEXT pixel is white.
const coords = tf.tensor2d(Array.from({ length: L }, (_, i) => [Math.floor(i / R) / R, (i % R) / R]));
const makeInput = (x, len) => tf.tidy(() => {           // x: [n, ≥len−1] pixels so far
  const n = x.shape[0];
  const prev = (len === 1 ? tf.zeros([n, 1]) : tf.concat([tf.zeros([n, 1]), x.slice([0, 0], [n, len - 1])], 1)).expandDims(2);
  return tf.concat([prev, tf.tile(coords.slice([0, 0], [len, 2]).expandDims(0), [n, 1, 1])], 2);
});
const model = tf.sequential({ layers: [
  tf.layers.gru({ units: P.units, returnSequences: true, inputShape: [null, 3] }),
  tf.layers.dense({ units: 1 }),                         // logit of "next pixel is white"
] });
const params = model.trainableWeights.map(w => w.val);
const opt = tf.train.adam(P.lr);

// ── STEP 3 · Train: maximise the likelihood of every pixel given its past ─
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => {
    const xb = lab.batch(bits, 64), inp = makeInput(xb, L), target = xb.expandDims(2);
    return opt.minimize(() => tf.losses.sigmoidCrossEntropy(target, model.apply(inp, { training: true })), true, params);
  });
  if (step % 5 === 0) {
    const v = tf.tidy(() => tf.losses.sigmoidCrossEntropy(testBits.expandDims(2), model.predict(makeInput(testBits, L))));
    lab.plot('train', (await cost.data())[0]); lab.plot('test', (await v.data())[0]); v.dispose();
  }
  cost.dispose();
  if (step % 100 === 0 || step === P.steps - 1) {
    const s = sample(16); lab.show(s, 'Generated pixel by pixel', { cols: 8, scale: 8, key: 'live' }); s.dispose();
  }
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 4 · Sampling: flip a biased coin for every pixel in order ─
// prefix (optional) = pixels we already know → the model completes the rest of the picture
function sample(n, prefix) {
  return tf.tidy(() => {
    let x = prefix || tf.zeros([n, 0]);
    for (let t = x.shape[1]; t < L; t++) {
      const logits = model.predict(makeInput(x, t + 1));
      const p = tf.sigmoid(logits.slice([0, t, 0], [n, 1, 1]).reshape([n, 1]));
      x = tf.concat([x, tf.randomUniform([n, 1]).less(p).cast('float32')], 1);
    }
    return x.reshape([n, R, R, 1]);
  });
}
const complete = (img, rows, n) => tf.tidy(() => {      // img: [1,R,R,1] bits; keep first `rows` rows
  const k = rows * R; return sample(n, tf.tile(img.reshape([1, L]).slice([0, 0], [1, k]), [n, 1]));
});
lab.register({ id: 'pixelrnn', label: 'PixelRNN (8×8)', sample: async n => sample(n) });
return { R, L, sample, complete, toBits: x => toBits(x).reshape([-1, R, R, 1]) };
