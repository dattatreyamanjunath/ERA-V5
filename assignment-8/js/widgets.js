/* ============================================================================
   widgets.js — the five calculators. Each one recomputes a specific claim from
   the session so a reader can move the inputs and watch the trade appear,
   rather than being told the conclusion.
   ========================================================================== */
(function () {
  'use strict';
  const $ = s => document.querySelector(s);
  const n = x => x.toLocaleString();

  /* ---------- 1. softmax on/off: two routes, one answer ---------- */
  const smToggle = $('#sm-toggle');
  function drawSoftmax() {
    const useSm = smToggle.checked;
    const q = 2, keys = [0.5, 1.0, 1.5], values = [10, 20, 30];
    const r = noSoftmaxDemo(q, keys, values, useSm);
    const rows = keys.map((k, i) =>
      `<div class="vecrow">k${i + 1} = ${k.toFixed(1)} &nbsp; v${i + 1} = ${values[i]} &nbsp;→&nbsp; score ${(q * k).toFixed(1)}` +
      (useSm ? ` &nbsp;→&nbsp; weight <b>${r.weights[i].toFixed(2)}</b>` : '') + `</div>`).join('');

    $('#softmax-demo').innerHTML = `
      <div class="vecrow" style="margin-bottom:8px">query q = ${q}</div>
      ${rows}
      <div class="vs" style="margin-top:14px">
        <div class="side ${useSm ? '' : 'right'}">
          <h4>Direct route</h4>
          <p class="small" style="margin:0">Visit every old key and value.</p>
          <div class="bigno">${r.direct.toFixed(useSm ? 1 : 0)}</div>
          <p class="tiny muted" style="margin:4px 0 0">Cost grows with how many old tokens there are.</p>
        </div>
        <div class="mid">${useSm ? '✗' : '='}</div>
        <div class="side ${useSm ? 'left' : 'right'}">
          <h4>Regrouped route</h4>
          ${useSm
            ? `<p class="small" style="margin:0">Not available.</p>
               <div class="bigno" style="color:var(--warn)">—</div>
               <p class="tiny muted" style="margin:4px 0 0">The softmax denominator depends on this query, so q cannot be factored out.</p>`
            : `<p class="small" style="margin:0">Read one pre-built state S = Σ kⱼvⱼ = <b>${r.state}</b>.</p>
               <div class="bigno">${r.regrouped.toFixed(0)}</div>
               <p class="tiny muted" style="margin:4px 0 0">S is the same size after 10 tokens or 1,000,000.</p>`}
        </div>
      </div>
      <p class="small" style="margin-top:12px;color:${useSm ? 'var(--warn)' : 'var(--ok)'}">
        ${useSm ? r.note : (r.matches ? '✓ Both routes return exactly ' + r.direct + '. ' : '') + r.note}
      </p>`;
  }
  smToggle.addEventListener('change', drawSoftmax);
  drawSoftmax();

  /* ---------- 2. the delta rule ---------- */
  const dWant = $('#delta-want');
  function drawDelta() {
    const want = +dWant.value;
    $('#delta-wantval').textContent = want;
    const d = deltaRuleDemo(40, want);
    const wrong = d.addOnlyResult !== d.wantedAnswer;
    $('#delta-demo').innerHTML = `
      <div class="vs">
        <div class="side left">
          <h4>Add-only write</h4>
          <p class="small" style="margin:0">new state = old state + whole new answer</p>
          <div class="vecrow" style="margin-top:6px">${d.oldAnswer} + ${d.wantedAnswer}</div>
          <div class="bigno" style="color:var(--warn)">${d.addOnlyResult}</div>
          <p class="tiny" style="margin:4px 0 0;color:var(--warn)">
            ${wrong ? `wanted ${d.wantedAnswer} — the old contribution is still in there` : 'happens to be right only because the delta equals the answer'}
          </p>
        </div>
        <div class="mid">vs</div>
        <div class="side right">
          <h4>Delta rule</h4>
          <p class="small" style="margin:0">read first, then write only the difference</p>
          <div class="vecrow" style="margin-top:6px">Δ = ${d.wantedAnswer} − ${d.oldAnswer} = <b>${d.delta}</b></div>
          <div class="bigno" style="color:var(--ok)">${d.deltaResult}</div>
          <p class="tiny muted" style="margin:4px 0 0">${d.oldAnswer} + ${d.delta} = ${d.deltaResult}</p>
        </div>
      </div>
      <p class="small muted" style="margin-top:12px">
        There is only ever <em>one</em> state matrix — the fixed size was never the problem. The problem is
        that an add-only rule has no term that cancels what is no longer wanted. That single missing
        subtraction is why a fixed-size memory needs the delta rule to be useful at all.
      </p>`;
  }
  dWant.addEventListener('input', drawDelta);
  drawDelta();

  /* ---------- 3. the KV cache bill ---------- */
  const kvIn = {
    layers: $('#kv-layers'), heads: $('#kv-heads'), dim: $('#kv-dim'),
    t: $('#kv-t'), batch: $('#kv-batch'), bytes: $('#kv-bytes')
  };
  function drawKV() {
    const layers = +kvIn.layers.value, kvHeads = +kvIn.heads.value, headDim = +kvIn.dim.value;
    const tokens = +kvIn.t.value, batch = +kvIn.batch.value, bpn = +kvIn.bytes.value;
    $('#kv-layersv').textContent = layers; $('#kv-headsv').textContent = kvHeads;
    $('#kv-dimv').textContent = headDim; $('#kv-tv').textContent = n(tokens);
    $('#kv-batchv').textContent = batch;

    const perUser = kvCacheBytes({ layers, kvHeads, headDim, tokens, batch: 1, bytesPerNumber: bpn });
    const total = perUser * batch;
    const perToken = kvCacheBytes({ layers, kvHeads, headDim, tokens: 1, batch: 1, bytesPerNumber: bpn });

    $('#kv-demo').innerHTML = `
      <div class="kv">
        <div class="cell">2 <span class="muted">key and value</span></div>
        <div class="cell">× ${layers} <span class="muted">layers</span></div>
        <div class="cell">× ${kvHeads} <span class="muted">kv_heads</span></div>
        <div class="cell">× ${headDim} <span class="muted">head_dim</span></div>
        <div class="cell">× ${n(tokens)} <span class="muted">tokens</span></div>
        <div class="cell">× ${batch} <span class="muted">users</span></div>
        <div class="cell">× ${bpn} <span class="muted">bytes</span></div>
      </div>
      <div class="grid3">
        <div><div class="bigno">${fmtBytes(perToken)}</div><span class="small muted">added by each new token</span></div>
        <div><div class="bigno">${fmtBytes(perUser)}</div><span class="small muted">one conversation</span></div>
        <div><div class="bigno">${fmtBytes(total)}</div><span class="small muted">${batch} concurrent user${batch === 1 ? '' : 's'}</span></div>
      </div>
      <p class="small muted" style="margin:12px 0 0">
        Context length sets the cost of one conversation; concurrency multiplies it. Model weights load
        once and are shared across every user — this is not, which is what makes it a serving cost rather
        than a model cost. Counts the raw K and V tensors only: a real server also needs weights,
        activations, attention workspace and allocator headroom.
      </p>`;
  }
  Object.values(kvIn).forEach(el => el.addEventListener('input', drawKV));
  drawKV();

  /* ---------- 4. MHA / GQA / MQA ---------- */
  const GQA_MODES = [
    { id: 'mha', label: 'MHA — 8 K/V heads', kv: 8 },
    { id: 'gqa4', label: 'GQA — 4 K/V heads', kv: 4 },
    { id: 'gqa2', label: 'GQA — 2 K/V heads', kv: 2 },
    { id: 'mqa', label: 'MQA — 1 K/V head', kv: 1 }
  ];
  let gqaMode = 'gqa2';
  $('#gqa-controls').innerHTML = GQA_MODES.map(m =>
    `<button class="chipbtn" data-gqa="${m.id}" type="button" aria-pressed="${m.id === gqaMode}">${m.label}</button>`).join('');
  $('#gqa-controls').addEventListener('click', e => {
    const b = e.target.closest('[data-gqa]'); if (!b) return;
    gqaMode = b.dataset.gqa;
    document.querySelectorAll('[data-gqa]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.gqa === gqaMode)));
    drawGQA();
  });
  function drawGQA() {
    const mode = GQA_MODES.find(m => m.id === gqaMode);
    const groupSize = 8 / mode.kv;
    const palette = ['var(--memory)', 'var(--compute)', 'var(--position)', 'var(--quality)',
                     '#d98cb3', '#8ec9a5', '#c9a15f', '#8fa0d6'];
    const heads = Array.from({ length: 8 }, (_, i) => {
      const g = Math.floor(i / groupSize);
      return `<div class="qhead" style="background:${palette[g % palette.length]}" title="query head ${i + 1} → K/V head ${g + 1}">Q${i + 1}</div>`;
    }).join('');
    const kvheads = Array.from({ length: mode.kv }, (_, g) =>
      `<div class="qhead" style="background:${palette[g % palette.length]};width:${30 * groupSize + 4 * (groupSize - 1)}px">KV${g + 1}</div>`).join('');

    const at32k = kvCacheBytes({ layers: 48, kvHeads: mode.kv, headDim: 128, tokens: 32768, batch: 1, bytesPerNumber: 2 });
    const at1m = kvCacheBytes({ layers: 48, kvHeads: mode.kv, headDim: 128, tokens: 1048576, batch: 1, bytesPerNumber: 2 });
    const mhaAt1m = kvCacheBytes({ layers: 48, kvHeads: 8, headDim: 128, tokens: 1048576, batch: 1, bytesPerNumber: 2 });

    $('#gqa-demo').innerHTML = `
      <div class="small muted" style="margin-bottom:4px">query heads</div>
      <div class="headviz">${heads}</div>
      <div class="small muted" style="margin:8px 0 4px">shared key/value heads</div>
      <div class="headviz">${kvheads}</div>
      <div class="kv" style="margin-top:14px">
        <div class="cell">at 32K: <b>${fmtBytes(at32k)}</b></div>
        <div class="cell">at 1M: <b>${fmtBytes(at1m)}</b></div>
        <div class="cell">vs MHA: <b>${(mhaAt1m / at1m).toFixed(0)}× smaller</b></div>
      </div>
      <p class="small muted" style="margin:0">
        ${mode.kv === 8
          ? 'Every query head gets its own keys and values — maximum diversity, maximum cache.'
          : mode.kv === 1
            ? 'One K/V head for everything. The largest saving, and the one with a real quality cost: all eight heads must now agree on a single view of the past.'
            : 'Query heads still ask different questions; there are simply fewer stored versions of the answers to search.'}
        Either way the cache at 1M is still <b>${fmtBytes(at1m)}</b> — sharing lowers the line, it does not
        stop it climbing. That is why GQA is a baseline and not an answer.
      </p>`;
  }
  drawGQA();

  /* ---------- 5. sequence compression + top-k ---------- */
  const cmpM = $('#cmp-m'), cmpK = $('#cmp-k');
  function drawCompress() {
    const T = 64, m = +cmpM.value, k = Math.min(+cmpK.value, Math.ceil(T / m));
    $('#cmp-mv').textContent = m; $('#cmp-kv').textContent = k;
    const blocks = Math.ceil(T / m);
    // deterministic "relevance" so the picture is stable between renders
    const score = i => Math.abs(Math.sin(i * 2.399)) * (0.55 + 0.45 * (i / blocks));
    const ranked = Array.from({ length: blocks }, (_, i) => [i, score(i)])
      .sort((a, b) => b[1] - a[1]).slice(0, k).map(x => x[0]);
    const sel = new Set(ranked);

    const boxes = Array.from({ length: blocks }, (_, i) =>
      `<div class="tokbox ${sel.has(i) ? 'sel' : 'comp'}" style="width:${Math.max(14, Math.min(40, m * 7))}px"
            title="block ${i + 1}${sel.has(i) ? ' — selected' : ''}"></div>`).join('');

    $('#compress-demo').innerHTML = `
      <div class="small muted">${T} tokens → ${blocks} stored entries (${m} token${m === 1 ? '' : 's'} each), of which ${k} are read</div>
      <div class="blockviz">${boxes}</div>
      <div class="kv">
        <div class="cell">stored: ${T} → <b>${blocks}</b> <span class="muted">(${(T / blocks).toFixed(1)}× less)</span></div>
        <div class="cell">read by expensive attention: <b>${k}</b> <span class="muted">of ${blocks}</span></div>
        <div class="cell">end to end: <b>${(T / k).toFixed(1)}×</b> <span class="muted">fewer reads than dense</span></div>
      </div>
      <p class="small muted" style="margin:0">
        These are two <em>different</em> savings and they are often conflated. Compression cuts what you
        <b>store</b>; top-k cuts what you <b>read</b>. Compression is lossy — one entry now speaks for ${m}
        token${m === 1 ? '' : 's'}. And top-k only helps if the ranking is cheaper than scoring everything,
        which is why DeepSeek uses a small low-rank indexer to choose the blocks rather than full attention.
      </p>`;
  }
  cmpM.addEventListener('input', drawCompress);
  cmpK.addEventListener('input', drawCompress);
  drawCompress();

  /* ---------- 6. depth schedules ---------- */
  const SCHEDULES = [
    { id: 'allg', label: 'G every layer', pat: 'GGGGGGGG', note: 'Exact token access in every layer, and a KV cache in every layer. Maximum quality, maximum serving memory.' },
    { id: 'v4', label: 'D D D G D D D G', pat: 'DDDGDDDG', note: 'The LightningLM V4 motif, kept from the 1.78B seed model all the way to the 120B run. That consistency is real evidence the mixture works across scale — it is not evidence that 6:2 is the optimal ratio, because the neighbouring schedules were never cleanly ablated.' },
    { id: 'qwen', label: 'D D D G (3:1)', pat: 'DDDGDDDG', note: 'The same 3:1 ratio Qwen3-Next and Kimi Linear both ship. Two labs, two architectures, the same number — which is either converging evidence or shared inheritance. Nobody has run the experiment that would tell you which.' },
    { id: 'alld', label: 'D every layer', pat: 'DDDDDDDD', note: 'No KV cache at all and the cheapest possible serving. But the model never gets another direct look at exact earlier tokens — everything it knows about the past is a compressed summary.' }
  ];
  let schedMode = 'v4';
  $('#sched-controls').innerHTML = SCHEDULES.map(s =>
    `<button class="chipbtn" data-sched="${s.id}" type="button" aria-pressed="${s.id === schedMode}">${s.label}</button>`).join('');
  $('#sched-controls').addEventListener('click', e => {
    const b = e.target.closest('[data-sched]'); if (!b) return;
    schedMode = b.dataset.sched;
    document.querySelectorAll('[data-sched]').forEach(x => x.setAttribute('aria-pressed', String(x.dataset.sched === schedMode)));
    drawSched();
  });
  function drawSched() {
    const s = SCHEDULES.find(x => x.id === schedMode);
    const gCount = (s.pat.match(/G/g) || []).length;
    const baseG = 1; // one G layer per 8 as the reference point
    const cacheX = gCount / baseG;
    // mixing compute scales far more gently than cache; the session reports
    // 8x cache for ~1.41x compute going from 1 G-layer per 8 to 8 per 8.
    const computeX = Math.pow(cacheX, Math.log(1.41) / Math.log(8));

    $('#sched-demo').innerHTML = `
      <div class="schedviz">${s.pat.split('').map(c =>
        `<div class="slot ${c}" title="${c === 'D' ? 'DeltaNet fixed-state layer' : 'sparse-attention layer'}">${c}</div>`).join('')}</div>
      <div class="kv">
        <div class="cell">sparse-attention layers: <b>${gCount}</b> of 8</div>
        <div class="cell">KV state vs 1-per-8: <b>${cacheX.toFixed(1)}×</b></div>
        <div class="cell">mixing compute vs 1-per-8: <b>${computeX.toFixed(2)}×</b></div>
      </div>
      <p class="small muted" style="margin:0 0 8px">${s.note}</p>
      <p class="small muted" style="margin:0">
        The asymmetry is the point: going from one G layer per eight to all eight makes the KV state
        <b>8.0×</b> larger while mixing compute rises only about <b>1.41×</b>. In this configuration the
        price of more exact access is serving <em>memory</em>, not FLOPs — so if you were optimising for
        FLOPs you would reach the wrong conclusion about the ratio.
      </p>`;
  }
  drawSched();
})();
