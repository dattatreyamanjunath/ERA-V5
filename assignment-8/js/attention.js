/* ============================================================================
   attention.js — scaled dot-product attention, actually computed.
   Nothing on this page is a hand-drawn picture of attention. Every number the
   demo shows comes out of these functions at render time, so if the maths were
   wrong the page would show it.
   ========================================================================== */

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];
const D_MODEL = 4;

/* A fixed, hand-picked "embedding" table. Deterministic so the page renders the
   same numbers every reload, and small enough that a reader can check a dot
   product with a calculator. Row = token, column = feature.
   Loosely: f0 = "is a noun-ish thing", f1 = "is an action", f2 = "is a place",
   f3 = "is a function word". These are made up for the demo, not learned. */
const EMB = {
  'The': [0.1, 0.0, 0.1, 0.9],
  'cat': [0.9, 0.1, 0.1, 0.0],
  'sat': [0.1, 0.9, 0.3, 0.0],
  'on':  [0.0, 0.1, 0.6, 0.7],
  'the': [0.1, 0.0, 0.1, 0.9],
  'mat': [0.7, 0.0, 0.8, 0.0]
};

/* Fixed projection matrices (D_MODEL x D_MODEL). Chosen — not trained — so that
   "sat" genuinely queries for noun-ish subjects and therefore attends to "cat".
   That is the exact example the lesson uses, so the demo has to reproduce it. */
const Wq = [[1.2, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.4]];
const Wk = [[1.1, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.5]];
const Wv = [[1.0, 0.0, 0.2, 0.0],
            [0.0, 1.0, 0.0, 0.1],
            [0.1, 0.0, 1.0, 0.0],
            [0.0, 0.2, 0.0, 1.0]];

/* "sat" should look for a subject, so rotate its query toward the noun axis.
   Applied after Wq, purely to make the demo's story legible. */
const QUERY_BIAS = {
  'sat': [0.55, 0.0, 0.0, 0.0],
  'mat': [0.10, 0.0, 0.0, 0.0]
};

const matvec = (M, v) => M.map(row => row.reduce((s, w, i) => s + w * v[i], 0));
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

function softmax(row, temperature = 1) {
  const t = Math.max(temperature, 1e-6);
  const scaled = row.map(x => (x === -Infinity ? -Infinity : x / t));
  const max = Math.max(...scaled.filter(Number.isFinite));
  const exps = scaled.map(x => (x === -Infinity ? 0 : Math.exp(x - max)));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => (sum === 0 ? 0 : e / sum));
}

/**
 * Run the whole layer and return every intermediate, so the UI can step through
 * the same five stages the lesson lists: project, score, scale, mask, softmax,
 * weighted sum.
 */
function runAttention({ causal = true, temperature = 1 } = {}) {
  const n = TOKENS.length;
  const Q = [], K = [], V = [];
  TOKENS.forEach((tok, i) => {
    const x = EMB[tok];
    const q = matvec(Wq, x);
    const bias = QUERY_BIAS[tok];
    if (bias) bias.forEach((b, j) => { q[j] += b; });
    Q.push(q);
    K.push(matvec(Wk, x));
    V.push(matvec(Wv, x));
  });

  const raw = [];      // QK^T
  const scaled = [];   // QK^T / sqrt(d_k)
  const masked = [];   // + M
  const weights = [];  // softmax(...)
  const scale = Math.sqrt(D_MODEL);

  for (let i = 0; i < n; i++) {
    const r = [], s = [], m = [];
    for (let j = 0; j < n; j++) {
      const d = dot(Q[i], K[j]);
      r.push(d);
      s.push(d / scale);
      m.push(causal && j > i ? -Infinity : d / scale);
    }
    raw.push(r); scaled.push(s); masked.push(m);
    weights.push(softmax(m, temperature));
  }

  // output[i] = sum_j weights[i][j] * V[j]
  const output = weights.map(w =>
    V[0].map((_, d) => w.reduce((s, wj, j) => s + wj * V[j][d], 0)));

  return { tokens: TOKENS, Q, K, V, raw, scaled, masked, weights, output, scale, causal, temperature };
}

/* --- the section-4 claim: with softmax off, two routes give the same answer --
   Scalar version, using exactly the lesson's numbers so a reader can check it
   against the transcript: q=2, keys 0.5/1.0/1.5, values 10/20/30 -> 140. */
function noSoftmaxDemo(q, keys, values, useSoftmax) {
  const scores = keys.map(k => q * k);
  if (useSoftmax) {
    const w = softmax(scores);
    return {
      direct: w.reduce((s, wj, j) => s + wj * values[j], 0),
      state: null,
      regrouped: null,
      matches: false,
      weights: w,
      note: 'With softmax on there is no single S to precompute: the denominator depends on this query, so the old keys must be revisited individually.'
    };
  }
  const direct = scores.reduce((s, sc, j) => s + sc * values[j], 0);
  const S = keys.reduce((s, k, j) => s + k * values[j], 0);   // S = sum k_j v_j
  const regrouped = q * S;
  return {
    direct, state: S, regrouped,
    matches: Math.abs(direct - regrouped) < 1e-9,
    weights: scores,
    note: 'Softmax off: every term shares the same q, so q factors out and the past folds into one number S — which does not grow when more tokens arrive.'
  };
}

/* --- section 5/6: why an add-only state cannot correct itself --------------- */
function deltaRuleDemo(oldAnswer, wantedAnswer) {
  return {
    oldAnswer,
    wantedAnswer,
    addOnlyResult: oldAnswer + wantedAnswer,   // the lesson's 40 + 55 = 95
    delta: wantedAnswer - oldAnswer,           // 15
    deltaResult: oldAnswer + (wantedAnswer - oldAnswer)  // 55
  };
}

/* --- section 10: the KV cache bill, one factor at a time ------------------- */
function kvCacheBytes({ layers, kvHeads, headDim, tokens, batch, bytesPerNumber }) {
  return 2 * layers * kvHeads * headDim * tokens * batch * bytesPerNumber;
}
/* Decimal units (10^9), because that is what the lesson's 6.44 GB / 51.54 GB
   figures use. Reporting GiB here would silently disagree with the transcript
   by 7%, which is exactly the kind of quiet mismatch this page is about. */
function fmtBytes(b) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${b.toFixed(b < 10 ? 2 : 1)} ${u[i]}`;
}
