/* ============================================================================
   timeline.js — the chronological axis, the detail drawer, and the live
   scaled-dot-product demo. Everything here reads from MECHANISMS; nothing
   about a mechanism is written twice.
   ========================================================================== */
(function () {
  'use strict';
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  /* long raw URLs are unbreakable and blow out the mobile layout, so links get a
     readable label and keep the full URL in href */
  const shortUrl = (u) => {
    try {
      const p = new URL(u);
      const path = p.pathname.replace(/\/$/, '');
      return p.host.replace(/^www\./, '') + (path.length > 34 ? path.slice(0, 34) + '…' : path);
    } catch { return u; }
  };

  /* ---------------- theme ---------------- */
  const themeBtn = $('#themebtn');
  const stored = localStorage.getItem('attn-theme');
  if (stored) document.documentElement.setAttribute('data-theme', stored);
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const isDark = cur ? cur === 'dark'
      : !window.matchMedia('(prefers-color-scheme: light)').matches;
    const next = isDark ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('attn-theme', next);
    drawBills();
  });

  /* ---------------- hero stats ---------------- */
  const years = MECHANISMS[MECHANISMS.length - 1].date.slice(0, 4) - MECHANISMS[0].date.slice(0, 4);
  $('#stat-count').textContent = MECHANISMS.length;
  $('#stat-span').textContent = years + '½';
  $('#stat-verified').textContent = MECHANISMS.filter(m => m.arxiv).length + '/' + MECHANISMS.length;
  $('#stat-eras').textContent = ERAS.length;

  /* ---------------- legend / filtering ---------------- */
  const active = new Set(Object.keys(PAYS));
  const legend = $('#tl-legend');
  legend.innerHTML = Object.entries(PAYS).map(([k, v]) =>
    `<button class="legchip" data-pays="${k}" aria-pressed="true" type="button">
       <span class="dot" style="background:var(--${k})"></span>${esc(v.label)}
     </button>`).join('') +
    `<span class="tiny muted" style="align-self:center;margin-left:6px">
       click to filter by which bill a mechanism pays down</span>`;

  legend.addEventListener('click', (e) => {
    const btn = e.target.closest('.legchip'); if (!btn) return;
    const k = btn.dataset.pays;
    if (active.has(k)) { active.delete(k); btn.setAttribute('aria-pressed', 'false'); }
    else { active.add(k); btn.setAttribute('aria-pressed', 'true'); }
    if (active.size === 0) { Object.keys(PAYS).forEach(x => active.add(x)); $$('.legchip').forEach(b => b.setAttribute('aria-pressed', 'true')); }
    applyFilter();
  });
  function applyFilter() {
    $$('.node').forEach(n => {
      const pays = n.dataset.pays.split(',');
      n.classList.toggle('dim', !pays.some(p => active.has(p)));
    });
  }

  /* ---------------- timeline ---------------- */
  const root = $('#timeline-root');
  root.innerHTML = ERAS.map(era => {
    const items = MECHANISMS.filter(m => m.date >= era.from && m.date <= era.to);
    if (!items.length) return '';
    const y1 = era.from.slice(0, 4), y2 = era.to.slice(0, 4);
    return `<div class="era">
      <div class="era-head">
        <span class="yrs">${y1}${y1 === y2 ? '' : '–' + y2}</span>
        <h3>${esc(era.label)}</h3>
      </div>
      <p class="era-blurb">${esc(era.blurb)}</p>
      ${items.map(nodeHTML).join('')}
    </div>`;
  }).join('');

  function nodeHTML(m) {
    const primary = m.pays[0];
    const tags = m.pays.map(p =>
      `<span class="tag" style="color:var(--${p});border-color:var(--${p})">${esc(PAYS[p].label)}</span>`).join('');
    return `<div class="node ${m.anchor ? 'is-anchor' : ''}" data-id="${m.id}" data-pays="${m.pays.join(',')}"
                 style="--dotcolor:var(--${primary})">
      <button class="node-btn" type="button">
        <div class="node-top">
          <span class="node-date">${esc(prettyDate(m.date))}</span>
          <span class="node-name">${esc(m.name)}</span>
          ${m.required ? '<span class="badge-req">in brief</span>' : ''}
          ${m.flagged ? '<span class="badge-req badge-flag">see corrections</span>' : ''}
          <span class="node-tags">${tags}</span>
        </div>
        <div class="node-problem">${esc(m.problem)}</div>
        <div class="datechip">
          <span class="k">source</span> ${esc(m.sourceLabel)}<br>
          <span class="k">date basis</span> ${esc(m.dateBasis)}
        </div>
      </button>
    </div>`;
  }

  /* ---------------- verify mode ---------------- */
  const nPaper = MECHANISMS.filter(m => m.arxiv).length;
  $('#verify-summary').innerHTML =
    `<b>${nPaper}</b> of ${MECHANISMS.length} dates are arXiv v1 submission dates, re-checked
     against the arXiv API. The other ${MECHANISMS.length - nPaper} have no paper and are
     labelled individually.`;
  $('#verify-toggle').addEventListener('change', e => {
    document.body.classList.toggle('verify', e.target.checked);
  });

  /* weak-dates list in the corrections section */
  $('#weak-dates').innerHTML = MECHANISMS.filter(m => !m.arxiv).map(m => `
    <div class="tradebox" style="margin-bottom:10px">
      <h4 style="color:var(--ink-2)">${esc(prettyDate(m.date))} · ${esc(m.name)}</h4>
      <p class="small">${esc(m.dateBasis)}</p>
      <p class="small" style="margin-top:6px"><a href="${esc(m.sourceUrl)}" target="_blank" rel="noopener">${esc(shortUrl(m.sourceUrl))}</a></p>
    </div>`).join('');

  /* ---------------- drawer ---------------- */
  const drawer = $('#drawer'), scrim = $('#scrim'), body = $('#drawer-body');
  let lastFocus = null;

  root.addEventListener('click', e => {
    const node = e.target.closest('.node'); if (!node) return;
    lastFocus = node.querySelector('.node-btn');
    openDrawer(MECHANISMS.find(m => m.id === node.dataset.id));
  });

  function openDrawer(m) {
    const idx = MECHANISMS.indexOf(m);
    const prev = MECHANISMS[idx - 1], next = MECHANISMS[idx + 1];
    body.innerHTML = `
      <div class="eyebrow">${esc(prettyDate(m.date))}${m.arxiv ? ' · arXiv:' + esc(m.arxiv) : ' · no paper'}</div>
      <h2>${esc(m.name)}</h2>
      <div style="margin-bottom:14px">${m.pays.map(p =>
        `<span class="tag" style="color:var(--${p});border-color:var(--${p})">${esc(PAYS[p].label)}</span>`).join(' ')}</div>

      <div class="tradebox">
        <h4 style="color:var(--ink-2)">The problem at that moment</h4>
        <p>${esc(m.problem)}</p>
      </div>
      <div class="tradebox">
        <h4 style="color:var(--ink-2)">What it actually does</h4>
        <p>${esc(m.mechanism)}</p>
      </div>
      <div class="tradebox buys"><h4>What it buys</h4><p>${esc(m.buys)}</p></div>
      <div class="tradebox gives"><h4>What it gives up</h4><p>${esc(m.givesUp)}</p></div>
      <div class="tradebox pick"><h4>When you would actually pick it</h4><p>${esc(m.pickWhen)}</p></div>

      <div class="srcbox">
        <span class="lbl">Primary source</span>
        <p><a href="${esc(m.sourceUrl)}" target="_blank" rel="noopener">${esc(m.sourceLabel)}</a></p>
        <span class="lbl">How this date was established</span>
        <p>${esc(m.dateBasis)}</p>
      </div>

      <div style="display:flex;gap:8px;margin-top:18px;flex-wrap:wrap">
        ${prev ? `<button class="chipbtn" data-goto="${prev.id}" type="button">← ${esc(prev.short)}</button>` : ''}
        ${next ? `<button class="chipbtn" data-goto="${next.id}" type="button">${esc(next.short)} →</button>` : ''}
      </div>`;
    body.querySelectorAll('[data-goto]').forEach(b =>
      b.addEventListener('click', () => openDrawer(MECHANISMS.find(x => x.id === b.dataset.goto))));
    drawer.classList.add('open'); scrim.classList.add('open');
    drawer.setAttribute('aria-hidden', 'false');
    drawer.scrollTop = 0; drawer.focus();
  }
  function closeDrawer() {
    drawer.classList.remove('open'); scrim.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
    if (lastFocus) lastFocus.focus();
  }
  $('#drawer-close').addEventListener('click', closeDrawer);
  scrim.addEventListener('click', closeDrawer);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDrawer(); });

  /* ================= live scaled dot-product demo ================= */
  let stage = 4;
  const stageInfo = [
    ['Project every token into a query, a key and a value',
     'The same vector x goes through three different learned matrices. Looking for information and advertising information are different jobs, so they get different projections. Pick a row on the left to see its three vectors.'],
    ['Score — compare every query with every key',
     'One dot product per pair. Six tokens give 36 numbers. This grid <em>is</em> the T² bill: it is the only thing on this page that grows quadratically, and every mechanism after 2019 is trying to avoid filling it in.'],
    ['Scale — divide by √dₖ',
     'Here dₖ = 4, so every score is divided by 2. As the head dimension grows, raw dot products grow with it and push softmax into saturation, where gradients vanish. The scale factor is the entire fix.'],
    ['Mask — set every future score to −∞',
     'Grey cells are forbidden. Adding −∞ before the softmax means those positions receive exactly zero weight, so the model can be trained on the whole sequence in parallel while still generating strictly left to right. Untick the box and watch weight leak into the future.'],
    ['Softmax — turn each row into weights that sum to 1',
     'Every row now sums to exactly 1.0. Note the shared denominator: to compute the weight for key 1 you need the scores of keys 2 and 3. That coupling is precisely why exact attention must keep every individual old key — and why removing softmax lets the past fold into a fixed-size state.'],
    ['Weighted sum — combine the values',
     'Each output is the weighted average of the value vectors, using that row of weights. Row "sat" puts the most weight on "cat", which is the point: knowing <em>who</em> sat is what the query was asking for.']
  ];

  function heat(v, lo, hi, mode) {
    if (v === -Infinity) return { bg: 'var(--bg-3)', fg: 'var(--ink-3)' };
    const t = hi === lo ? 0.5 : (v - lo) / (hi - lo);
    if (mode === 'w') {
      const a = 0.10 + 0.90 * t;
      return { bg: `color-mix(in srgb, var(--accent) ${(a * 100).toFixed(0)}%, var(--bg-3))`,
               fg: t > 0.45 ? '#12161d' : 'var(--ink)' };
    }
    const a = 0.08 + 0.62 * t;
    return { bg: `color-mix(in srgb, var(--memory) ${(a * 100).toFixed(0)}%, var(--bg-3))`,
             fg: t > 0.55 ? '#12161d' : 'var(--ink)' };
  }

  function renderAttention() {
    const causal = $('#causal').checked;
    const temperature = parseFloat($('#temp').value);
    $('#tempval').textContent = temperature.toFixed(1);
    const r = runAttention({ causal, temperature });
    const [title, note] = stageInfo[stage];
    $('#attn-note').innerHTML = `<b>Step ${stage + 1} — ${title}.</b> ${note}`;
    $$('.step').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.stage === stage)));

    if (stage === 0) { renderQKV(r); return; }
    if (stage === 5) { renderOutput(r); return; }

    const grid = stage === 1 ? r.raw : stage === 2 ? r.scaled : stage === 3 ? r.masked : r.weights;
    const mode = stage === 4 ? 'w' : 's';
    const flat = grid.flat().filter(Number.isFinite);
    const lo = Math.min(...flat), hi = Math.max(...flat);

    let html = `<table class="matrix"><tr><th></th>` +
      r.tokens.map(t => `<th>${esc(t)}</th>`).join('') + `</tr>`;
    r.tokens.forEach((tok, i) => {
      html += `<tr><th class="rowh">${esc(tok)}</th>`;
      grid[i].forEach((v, j) => {
        const isMask = v === -Infinity || (stage === 4 && causal && j > i);
        const c = heat(v, lo, hi, mode);
        const txt = v === -Infinity ? '−∞' : v.toFixed(stage === 4 ? 3 : 2);
        html += `<td class="${isMask ? 'masked' : ''}" style="background:${c.bg};color:${c.fg}">${txt}</td>`;
      });
      html += `</tr>`;
    });
    html += `</table>`;
    $('#attn-matrix').innerHTML = html;

    if (stage === 4) {
      $('#attn-detail').innerHTML =
        `<div class="kv">` + r.tokens.map((t, i) =>
          `<div class="cell">${esc(t)} row Σ = <b>${r.weights[i].reduce((a, b) => a + b, 0).toFixed(4)}</b></div>`).join('') + `</div>` +
        (causal ? '' : `<p class="small" style="color:var(--warn);margin:0">
           Mask off: row "The" now puts ${(r.weights[0].slice(1).reduce((a, b) => a + b, 0) * 100).toFixed(0)}% of its
           weight on tokens that have not happened yet. At training time that is the model reading the answer.</p>`);
    } else {
      $('#attn-detail').innerHTML = stage === 2
        ? `<p class="small muted" style="margin:0">√dₖ = √4 = 2. Every cell above is the previous step's value halved.</p>`
        : '';
    }
  }

  function renderQKV(r) {
    let html = `<table class="matrix"><tr><th></th><th>x (embedding)</th></tr>`;
    r.tokens.forEach((tok, i) => {
      html += `<tr data-i="${i}"><th class="rowh">${esc(tok)}</th><td style="width:auto;background:var(--bg-3);color:var(--ink);padding:0 10px">` +
        r.Q[i].map((_, d) => '').join('') +
        `${fmtVec(EMB[tok])}</td></tr>`;
    });
    html += `</table>`;
    $('#attn-matrix').innerHTML = html;
    $('#attn-detail').innerHTML = r.tokens.map((tok, i) => `
      <div style="margin-bottom:9px">
        <div class="vecrow"><b>${esc(tok)}</b></div>
        <div class="vecrow" style="padding-left:12px">q = ${fmtVec(r.Q[i])}</div>
        <div class="vecrow" style="padding-left:12px">k = ${fmtVec(r.K[i])}</div>
        <div class="vecrow" style="padding-left:12px">v = ${fmtVec(r.V[i])}</div>
      </div>`).join('');
  }

  function renderOutput(r) {
    let html = `<table class="matrix"><tr><th></th><th>output vector</th></tr>`;
    r.tokens.forEach((tok, i) => {
      html += `<tr><th class="rowh">${esc(tok)}</th>
        <td style="width:auto;background:var(--bg-3);color:var(--ink);padding:0 10px">${fmtVec(r.output[i])}</td></tr>`;
    });
    html += `</table>`;
    $('#attn-matrix').innerHTML = html;
    const w = r.weights[2];
    const rank = r.tokens.map((t, j) => [t, w[j]]).filter(x => x[1] > 0).sort((a, b) => b[1] - a[1]);
    $('#attn-detail').innerHTML = `
      <p class="small" style="margin:0 0 8px"><b>Row "sat", in full:</b></p>
      <div class="kv">${rank.map(([t, v]) =>
        `<div class="cell">${esc(t)} <b>${(v * 100).toFixed(1)}%</b></div>`).join('')}</div>
      <p class="small muted" style="margin:0">
        output("sat") = ${rank.map(([t, v]) => `${v.toFixed(2)}·v(${esc(t)})`).join(' + ')}.
        The strongest contribution comes from <b>${esc(rank[0][0])}</b>.</p>`;
  }

  const fmtVec = v => '[' + v.map(x => x.toFixed(2).padStart(5)).join(', ') + ']';

  $$('.step').forEach(b => b.addEventListener('click', () => { stage = +b.dataset.stage; renderAttention(); }));
  $('#causal').addEventListener('change', renderAttention);
  $('#temp').addEventListener('input', renderAttention);
  renderAttention();

  /* ================= two bills chart ================= */
  const billsT = $('#bills-t');
  function drawBills() {
    const T = +billsT.value;
    $('#bills-tval').textContent = T.toLocaleString();
    $('#bills-pairs').textContent = (T * T).toLocaleString();
    $('#bills-cache').textContent = fmtBytes(kvCacheBytes(
      { layers: 48, kvHeads: 8, headDim: 128, tokens: T, batch: 1, bytesPerNumber: 2 }));

    const W = 520, H = 260, PL = 46, PR = 14, PT = 16, PB = 34;
    const maxT = 1048576;
    const x = t => PL + (t / maxT) * (W - PL - PR);
    const yq = t => H - PB - Math.pow(t / maxT, 2) * (H - PT - PB);
    const yl = t => H - PB - (t / maxT) * (H - PT - PB);
    const pts = f => Array.from({ length: 81 }, (_, i) => {
      const t = (i / 80) * maxT; return `${x(t).toFixed(1)},${f(t).toFixed(1)}`;
    }).join(' ');

    $('#bills-chart').innerHTML = `
      <line class="axis" x1="${PL}" y1="${H - PB}" x2="${W - PR}" y2="${H - PB}"/>
      <line class="axis" x1="${PL}" y1="${PT}" x2="${PL}" y2="${H - PB}"/>
      <polyline class="curve-compute" points="${pts(yq)}"/>
      <polyline class="curve-memory" points="${pts(yl)}"/>
      <line x1="${x(T)}" y1="${PT}" x2="${x(T)}" y2="${H - PB}" stroke="var(--ink-3)" stroke-dasharray="3 3"/>
      <circle cx="${x(T)}" cy="${yq(T)}" r="4.5" fill="var(--compute)"/>
      <circle cx="${x(T)}" cy="${yl(T)}" r="4.5" fill="var(--memory)"/>
      <text class="axislabel" x="${PL}" y="${H - 10}">0</text>
      <text class="axislabel" x="${W - PR}" y="${H - 10}" text-anchor="end">1M tokens</text>
      <text class="axislabel" x="${PL - 6}" y="${PT + 8}" text-anchor="end">cost</text>
      <text class="axislabel" x="${W - PR - 4}" y="${yq(maxT) + 14}" text-anchor="end" fill="var(--compute)">compute ∝ T²</text>
      <text class="axislabel" x="${W - PR - 4}" y="${yl(maxT) - 8}" text-anchor="end" fill="var(--memory)">KV cache ∝ T</text>`;
  }
  billsT.addEventListener('input', drawBills);
  drawBills();
  window.__drawBills = drawBills;
})();
