// ═══════════════════════════════════════════════════════════════
//  TRANSFORMER CAPTIONER · a tiny GPT that writes a caption for an image
// ═══════════════════════════════════════════════════════════════
// ── STEP 1 · Words become numbers ──────────────────────────────
const { S, C } = data;
const D = S * S * C, tok = data.tok, V = tok.vocab.length, Lc = tok.maxLen;
lab.log('vocabulary of ' + V + ' tokens, e.g. "' + data.captions.train[0] + '"');
const dm = P.dim, H = P.heads, dh = dm / H, NP = 4;      // model width, heads, number of image tokens
const flatImgs = data.xTrain.reshape([-1, D]);

// ── STEP 2 · Weights (embeddings, attention, feed-forward) ─────
const W = (shape, std = 0.05) => tf.variable(tf.randomNormal(shape, 0, std));
const Z0 = shape => tf.variable(tf.zeros(shape)), O1 = shape => tf.variable(tf.ones(shape));
const tokEmb = W([V, dm]), posEmb = W([NP + Lc, dm]), imgProj = W([D, NP * dm], 0.02), imgB = Z0([NP * dm]);
const blocks = Array.from({ length: P.layers }, () => ({ q: W([dm, dm]), k: W([dm, dm]), v: W([dm, dm]), o: W([dm, dm]),
  f1: W([dm, 4 * dm]), b1: Z0([4 * dm]), f2: W([4 * dm, dm]), b2: Z0([dm]), g1: O1([dm]), e1: Z0([dm]), g2: O1([dm]), e2: Z0([dm]) }));
const lnG = O1([dm]), lnB = Z0([dm]), head = W([dm, V]);
const params = [tokEmb, posEmb, imgProj, imgB, lnG, lnB, head, ...blocks.flatMap(b => Object.values(b))];

// ── STEP 3 · The transformer ───────────────────────────────────
const layerNorm = (x, g, b) => { const { mean, variance } = tf.moments(x, -1, true); return x.sub(mean).div(variance.add(1e-5).sqrt()).mul(g).add(b); };
const linear = (x, w, b) => { const [B, T, Din] = x.shape; let y = tf.matMul(x.reshape([B * T, Din]), w); if (b) y = y.add(b); return y.reshape([B, T, -1]); };
const gelu = x => x.mul(tf.sigmoid(x.mul(1.702)));
const attention = (x, blk) => {                          // causal self-attention: each token may only look LEFT
  const [B, T] = x.shape, split = t => t.reshape([B, T, H, dh]).transpose([0, 2, 1, 3]);
  const q = split(linear(x, blk.q)), k = split(linear(x, blk.k)), v = split(linear(x, blk.v));
  const mask = tf.sub(1, tf.linalg.bandPart(tf.ones([T, T]), -1, 0)).mul(-1e9);
  const att = tf.matMul(q, k, false, true).div(Math.sqrt(dh)).add(mask).softmax();   // who attends to whom
  return linear(tf.matMul(att, v).transpose([0, 2, 1, 3]).reshape([B, T, dm]), blk.o);
};
// sequence = [4 image tokens] + [<bos>, word, word, …].  Output at each word position predicts the NEXT word.
const run = (imgs, words) => {
  const B = imgs.shape[0], Lw = words.shape[1];
  const imgTok = linear(imgs.reshape([B, 1, D]), imgProj, imgB).reshape([B, NP, dm]);
  let x = tf.concat([imgTok, tf.gather(tokEmb, words)], 1).add(posEmb.slice([0, 0], [NP + Lw, dm]));
  for (const b of blocks) {
    x = x.add(attention(layerNorm(x, b.g1, b.e1), b));
    x = x.add(linear(gelu(linear(layerNorm(x, b.g2, b.e2), b.f1, b.b1)), b.f2, b.b2));
  }
  x = layerNorm(x, lnG, lnB);
  return linear(x.slice([0, NP, 0], [B, Lw, dm]), head);   // logits [B, Lw, V]
};

// ── STEP 4 · Loss: cross-entropy on the next word (ignore padding) ─
const lossFn = (imgs, caps) => {
  const inp = caps.slice([0, 0], [-1, Lc - 1]), target = caps.slice([0, 1], [-1, Lc - 1]);
  const logp = tf.logSoftmax(run(imgs, inp)), oh = tf.oneHot(target, V).cast('float32');
  const mask = target.notEqual(tok.PAD).cast('float32');
  return tf.neg(tf.sum(tf.mul(oh, logp), -1)).mul(mask).sum().div(mask.sum());
};
const opt = tf.train.adam(P.lr);
for (let step = 0; step < P.steps; step++) {
  const cost = tf.tidy(() => { const [x, c] = lab.batchXY(flatImgs, data.capTrain, 64); return opt.minimize(() => lossFn(x, c), true, params); });
  if (step % 10 === 0) lab.plot('caption loss', (await cost.data())[0]);
  cost.dispose();
  if (step % 200 === 0 || step === P.steps - 1) lab.log('step ' + step + ' · "' + caption(data.xTest.slice([0, 0, 0, 0], [1, S, S, C]), 0.6)[0] + '"  (true: "' + data.captions.test[0] + '")');
  if (!(await lab.tick(step, P.steps))) break;
}

// ── STEP 5 · Generation: sample one word at a time ─────────────
// temperature < 1 = safer wording · > 1 = more adventurous
function caption(x, temperature = 0.7) {
  const ids = tf.tidy(() => {
    const imgs = x.reshape([-1, D]), n = imgs.shape[0]; let words = tf.fill([n, 1], tok.BOS, 'int32');
    for (let j = 0; j < Lc - 1; j++) {
      const logits = run(imgs, words), last = logits.slice([0, words.shape[1] - 1, 0], [n, 1, V]).reshape([n, V]);
      words = tf.concat([words, tf.multinomial(last.div(temperature), 1).cast('int32')], 1);
    }
    return words;
  });
  const out = ids.arraySync().map(r => tok.decode(r.slice(1))); ids.dispose(); return out;
}
return { caption };
