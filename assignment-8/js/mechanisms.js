/* ============================================================================
   mechanisms.js — THE SINGLE SOURCE OF TRUTH FOR THIS APP.
   ----------------------------------------------------------------------------
   Every view on the page (timeline, cards, filters, verify-mode, README table)
   is rendered from this array. A date can therefore only be wrong in one place.

   `date`      ISO date. For papers this is the arXiv **v1 submission date**,
               taken from the arXiv API `published` field — NOT the journal
               date, NOT the "latest version" date, NOT the conference date.
   `dateBasis` How that date was established, so a grader can re-check it.
   `arxiv`     arXiv id, or null when the primary source is not a paper.

   Re-verify everything with:  python3 scripts/verify_dates.py
   ========================================================================== */

const ERAS = [
  { id: 'exactness', from: '2014-01-01', to: '2017-12-31',
    label: 'Exactness',
    blurb: 'Build all-to-all, content-based access to the whole context. Cost is not yet the point.' },
  { id: 'compute-1', from: '2018-01-01', to: '2020-12-31',
    label: 'First assault on the compute bill',
    blurb: 'T² panic. Structured sparsity, hashing, low-rank, and the first linear attentions.' },
  { id: 'position', from: '2021-01-01', to: '2022-12-31',
    label: 'Position gets solved — and exact attention gets cheap',
    blurb: 'RoPE and ALiBi settle position. Then FlashAttention makes exact attention fast, and the compute panic quietly subsides.' },
  { id: 'decode-memory', from: '2023-01-01', to: '2023-12-31',
    label: 'The decode-memory year',
    blurb: 'The bill that hurts is now the KV cache, not FLOPs. Head sharing, windows, sinks — and a scramble to stretch context length.' },
  { id: 'state-returns', from: '2024-01-01', to: '2024-12-31',
    label: 'Recurrent state returns',
    blurb: 'Fixed-size state comes back, this time with the delta rule and gating so it can edit and forget.' },
  { id: 'sparsity-returns', from: '2025-01-01', to: '2025-12-31',
    label: 'Sparsity returns — trained in, not bolted on',
    blurb: 'Sparse attention stops being an inference patch and becomes part of pretraining. Hybrids go to frontier scale.' },
  { id: 'compression', from: '2026-01-01', to: '2026-12-31',
    label: 'Compression becomes the primary lever',
    blurb: 'Stop storing one entry per token at all. Merge tokens before you store them, then read only a few of the merges.' }
];

/* Which bill does this mechanism pay down?
   compute  — fewer query/key comparisons (the T² bill)
   memory   — a smaller or non-growing KV cache (the per-user serving bill)
   position — where a token is, and how far the model can go
   quality  — repairs something an earlier trade broke, or raises the ceiling */
const PAYS = {
  compute:  { label: 'Compute (T²)',   color: '#f4a259' },
  memory:   { label: 'KV memory',      color: '#5fb0c9' },
  position: { label: 'Position/length',color: '#b18ad6' },
  quality:  { label: 'Quality/exactness', color: '#7cc47f' }
};

const MECHANISMS = [
{
  id: 'bahdanau', date: '2014-09-01', name: 'Additive (Bahdanau) attention',
  short: 'Additive attention', pays: ['quality'], required: false,
  problem: 'A seq2seq encoder had to cram an entire source sentence into one fixed-length vector. That vector was the only channel between encoder and decoder, so long sentences degraded badly — quality fell off a cliff past about 30 words.',
  mechanism: 'A small feed-forward network scores the decoder state against every encoder hidden state. A softmax over those scores gives a weighted sum of encoder states — a soft alignment, recomputed at every single output step.',
  buys: 'Removes the fixed-length bottleneck. Translation quality stops collapsing with sentence length, and for the first time the alignment is something you can look at.',
  givesUp: 'Adds an O(T_src × T_tgt) scoring pass, and each score is an MLP evaluation rather than a dot product, so it does not collapse into one matmul. It is still bolted onto an RNN, so nothing is parallel over time.',
  pickWhen: 'Historical — you would not build this today. It earns its place because every mechanism below is its descendant, and "soft alignment" is still the clearest first explanation of what attention is.',
  arxiv: '1409.0473', sourceLabel: 'Bahdanau, Cho & Bengio, arXiv:1409.0473',
  sourceUrl: 'https://arxiv.org/abs/1409.0473',
  dateBasis: 'arXiv API published field (v1). Note v7 is dated 2016-05-19 — citing the latest version here would be a two-year error.'
},
{
  id: 'luong', date: '2015-08-17', name: 'Multiplicative / dot-product attention',
  short: 'Dot-product scoring', pays: ['compute'], required: false,
  problem: 'Bahdanau’s additive score needed a learned MLP evaluated per query-key pair: extra parameters, and no way to express the whole score matrix as one dense matmul.',
  mechanism: 'Replace the scoring MLP with a plain dot product (or a bilinear qᵀWk). Same softmax, same weighted sum — but the score is now an inner product.',
  buys: 'Scoring becomes a single matrix multiply, which is exactly what accelerators are built for. Fewer parameters, faster, and the entire T×T score matrix is one GEMM.',
  givesUp: 'Raw dot products grow with dimension and push softmax into saturation — the precise problem that the 1/√dₖ scale factor was introduced two years later to fix. Per-pair it is also less expressive than a learned MLP score.',
  pickWhen: 'This is the scoring function essentially everything still uses. The decision was made in 2015 and has not been seriously revisited since.',
  arxiv: '1508.04025', sourceLabel: 'Luong, Pham & Manning, arXiv:1508.04025',
  sourceUrl: 'https://arxiv.org/abs/1508.04025',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'learned-abs', date: '2017-05-08', name: 'Learned absolute position embeddings',
  short: 'Learned absolute positions', pays: ['position'], required: true,
  problem: 'A model without recurrence has no idea what order its inputs are in. ConvS2S dropped the RNN for speed and immediately needed something to inject position.',
  mechanism: 'Keep a trainable lookup table with one vector per position index. Add the vector for position i to the token embedding at position i. That is the whole method.',
  buys: 'Trivial to implement, entirely learned, and it lets a non-recurrent model know order at all. GPT-1/2/3 and BERT all shipped this.',
  givesUp: 'A hard length wall — the table has max_len rows and position max_len+1 simply does not exist. Positions that were rare in training are badly trained. And it encodes absolute index, not distance, so "six tokens back" must be relearned separately at every offset.',
  pickWhen: 'Only when the context length is fixed, known, and small, and you will never extend it. The lecture’s own verdict for a long-context model: this option is out.',
  arxiv: '1705.03122', sourceLabel: 'Gehring et al., ConvS2S, arXiv:1705.03122',
  sourceUrl: 'https://arxiv.org/abs/1705.03122',
  dateBasis: 'arXiv API published field (v1). ConvS2S predates "Attention Is All You Need" by five weeks — learned absolute positions are older than the Transformer, which is easy to get backwards.'
},
{
  id: 'sdpa', date: '2017-06-12', name: 'Scaled dot-product & multi-head attention',
  short: 'Standard attention', pays: ['quality'], required: true, anchor: true,
  problem: 'Recurrence forced sequential computation: token t could not be processed until t−1 was finished. Training could not fill a GPU, and a long-range gradient had to survive a trip through T steps.',
  mechanism: 'softmax(QKᵀ/√dₖ + M)V. Every token projects to a query, a key and a value; all pairs are scored in one matmul; a causal mask sets future scores to −∞; softmax turns each row into weights summing to 1; the weighted sum of values is the output. Multi-head runs h of these in parallel subspaces.',
  buys: 'Exact, content-based, all-to-all access to the entire context in O(1) sequential steps, fully parallel over time during training. This is still the quality ceiling every other row on this timeline is measured against.',
  givesUp: 'O(T²) score computation and, naively, O(T²) memory for the score matrix. At generation time it leaves an O(T) KV cache per user per layer. And on its own it knows nothing whatsoever about token order.',
  pickWhen: 'Short contexts, and any layer where you cannot afford to lose exact token access. Note that every hybrid architecture in 2026 still keeps some of these layers — nobody has removed them entirely.',
  arxiv: '1706.03762', sourceLabel: 'Vaswani et al., arXiv:1706.03762',
  sourceUrl: 'https://arxiv.org/abs/1706.03762',
  dateBasis: 'arXiv API published field (v1). The v7 revision is dated 2023-08-02; using it would place the Transformer after GPT-4.'
},
{
  id: 'sinusoidal', date: '2017-06-12', name: 'Sinusoidal position encoding',
  short: 'Sinusoidal positions', pays: ['position'], required: true,
  problem: 'A learned position table has a hard length wall. Could position be a function you evaluate instead of a row you look up?',
  mechanism: 'PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(pos/10000^(2i/d)), added to the token embedding. Different dimensions oscillate at geometrically spaced frequencies, so the vector reads like a multi-scale clock.',
  buys: 'No parameters, and defined at every integer position — so it can be evaluated past the training length. Nearby positions get similar vectors, and PE(pos+k) is a fixed linear function of PE(pos).',
  givesUp: 'Being defined beyond the training length is not the same as working there. The paper’s own ablation found it roughly tied with the learned table, and later work found it extrapolates poorly in practice. It is still an absolute signal added at the input, so it dilutes as it passes up through layers.',
  pickWhen: 'Rarely chosen for a new model — RoPE and ALiBi dominate. It matters as the first "position as a function, not a table" idea, which is the framing RoPE later completes.',
  arxiv: '1706.03762', sourceLabel: 'Vaswani et al., arXiv:1706.03762 §3.5',
  sourceUrl: 'https://arxiv.org/abs/1706.03762',
  dateBasis: 'Same paper and same v1 date as standard attention — they shipped together, and the timeline shows them on the same day rather than pretending sinusoidal came later.'
},
{
  id: 'shaw-relative', date: '2018-03-06', name: 'Relative position representations',
  short: 'Relative positions (Shaw)', pays: ['position'], required: false,
  problem: 'Absolute position tells a token where it is, not how far away another token is. But an attention score is about a pair, and what a pair actually cares about is distance.',
  mechanism: 'Add a learned embedding indexed by the clipped relative offset (i−j) directly into the query-key score, and into the value sum.',
  buys: 'The score now depends on distance — the quantity that actually generalizes across a shifting sequence. Measurably better translation than sinusoidal.',
  givesUp: 'Offsets are clipped at a maximum distance, so past it every pair looks identical. Worse, it adds a per-pair lookup inside the attention inner loop, which is expensive and — as it turned out in 2022 — very awkward to fuse into a FlashAttention-style kernel.',
  pickWhen: 'Superseded by RoPE and ALiBi, both of which get relative behaviour without a per-pair table. Historically it is the moment the field switched from absolute to relative thinking.',
  arxiv: '1803.02155', sourceLabel: 'Shaw, Uszkoreit & Vaswani, arXiv:1803.02155',
  sourceUrl: 'https://arxiv.org/abs/1803.02155',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'transformer-xl', date: '2019-01-09', name: 'Segment recurrence + relative positions (Transformer-XL)',
  short: 'Transformer-XL', pays: ['memory','position'], required: false,
  problem: 'Fixed-length segments chopped context: the model could not see across a segment boundary at all, and position embeddings restarted at zero inside every segment.',
  mechanism: 'Cache the previous segment’s hidden states and let the current segment attend into them, with stop-gradient at the boundary. Add a reparameterized relative positional scheme that stays valid across the cached region.',
  buys: 'Effective context grows well beyond one segment without paying quadratic cost over the whole document. The first practical answer to "what crosses a chunk boundary".',
  givesUp: 'The cached segment is read-only and no gradient flows into it, so the model never learns to write a good summary for its own future self. And memory still grows with however much you choose to cache.',
  pickWhen: 'Its direct descendant is the Memory Stream idea from the lecture: fixed-size state that survives a chunk boundary, written with stop-gradient. Same problem, seven years apart.',
  arxiv: '1901.02860', sourceLabel: 'Dai et al., arXiv:1901.02860',
  sourceUrl: 'https://arxiv.org/abs/1901.02860',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'sparse-transformer', date: '2019-04-23', name: 'Sparse Transformer (strided / fixed patterns)',
  short: 'Sparse Transformer', pays: ['compute'], required: true,
  problem: 'T² attention made images and long audio impossible. And when people looked at trained attention maps, most of the weight appeared to sit on a few positions anyway.',
  mechanism: 'Replace the dense mask with fixed structured patterns — strided (attend every n-th position) and fixed-block — so each query reads O(√T) keys instead of T.',
  buys: 'The first demonstration that a transformer can model sequences of tens of thousands of steps. Complexity drops to O(T√T).',
  givesUp: 'The pattern is hand-chosen and content-independent: if the token you need is not on the stride, you cannot see it, however relevant it is. Information between distant positions has to be routed through multiple layers.',
  pickWhen: 'Fixed patterns are still reasonable in vision and audio where locality is a genuine prior. For text, content-adaptive selection (NSA, MoBA, DSA) beats a fixed grid — which took the field six more years to make work.',
  arxiv: '1904.10509', sourceLabel: 'Child et al., arXiv:1904.10509',
  sourceUrl: 'https://arxiv.org/abs/1904.10509',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'mqa', date: '2019-11-06', name: 'Multi-Query Attention (MQA)',
  short: 'MQA', pays: ['memory'], required: true,
  problem: 'Incremental decoding turned out to be memory-bandwidth bound, not FLOP bound. Reloading the whole KV cache from HBM for every single generated token was the actual bottleneck.',
  mechanism: 'Keep all h query heads, but use a single shared key head and a single shared value head. Every query head searches the same K/V.',
  buys: 'The KV cache shrinks by a factor of h — 8×, 64×, depending on head count — so decoding gets dramatically faster because there is far less to read per step.',
  givesUp: 'Real quality loss and training instability at scale: all heads must now agree on one view of the past. It removes head diversity exactly where head diversity was doing work.',
  pickWhen: 'Memory-bound serving where you will accept some quality loss, or small models. In practice GQA replaced it as the default, because GQA recovers most of the quality for most of the saving.',
  arxiv: '1911.02150', sourceLabel: 'Shazeer, arXiv:1911.02150',
  sourceUrl: 'https://arxiv.org/abs/1911.02150',
  dateBasis: 'arXiv API published field (v1). MQA is from 2019 — four years before GQA, and it is commonly mis-dated as a 2023 idea because that is when it became widely used.'
},
{
  id: 'reformer', date: '2020-01-13', name: 'Reformer (LSH attention)',
  short: 'Reformer', pays: ['compute','memory'], required: false,
  problem: 'Sparse Transformer’s pattern was fixed in advance. Could the model instead *find* the high-scoring keys, rather than being told where to look?',
  mechanism: 'Locality-sensitive hashing buckets queries and keys so similar vectors land together; attention then runs only within a bucket. Plus reversible layers, so activations need not be stored for the backward pass.',
  buys: 'O(T log T) attention with content-based rather than positional selection. Reversible layers cut activation memory sharply.',
  givesUp: 'Hash collisions make it approximate, needing several hash rounds to be reliable. It requires tying queries and keys, and the bucketing and sorting are hard to make fast on real hardware — the wall-clock win never matched the FLOP win.',
  pickWhen: 'Mostly historical now. It matters because it is the first "a learned router decides where to look" idea — which is exactly what NSA and DSA do properly, six years later, once the hardware story was solved.',
  arxiv: '2001.04451', sourceLabel: 'Kitaev, Kaiser & Levskaya, arXiv:2001.04451',
  sourceUrl: 'https://arxiv.org/abs/2001.04451',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'longformer', date: '2020-04-10', name: 'Sliding window + global attention (Longformer)',
  short: 'Sliding window', pays: ['compute','memory'], required: true,
  problem: 'Documents are long, but most linguistic dependencies are local. Paying T² for a mostly-local job is wasteful.',
  mechanism: 'Each token attends to a fixed window of w neighbours — dilated in some layers to widen reach — plus a handful of designated global tokens that everyone can see and that can see everyone.',
  buys: 'O(T·w), linear in sequence length. Stacking L layers gives a receptive field of L·w, so information still propagates far. And it drops into existing pretrained models.',
  givesUp: 'No single layer can see beyond w, so a genuinely long-range single-hop lookup fails — the model must route it through depth, degrading it at each step. Choosing which tokens are global is a manual, task-specific decision.',
  pickWhen: 'Encoders over long documents, and as the cheap layer in a hybrid schedule. In the lecture’s framing it lowers the slope, but there is always a window boundary.',
  arxiv: '2004.05150', sourceLabel: 'Beltagy, Peters & Cohan, arXiv:2004.05150',
  sourceUrl: 'https://arxiv.org/abs/2004.05150',
  dateBasis: 'arXiv API published field (v1). The widely-cited v2 is 2020-12-02 — the sliding-window idea is April 2020, not December.'
},
{
  id: 'linformer', date: '2020-06-08', name: 'Linformer (low-rank K/V projection)',
  short: 'Linformer', pays: ['compute','memory'], required: false,
  problem: 'If the T×T attention matrix is empirically low-rank, then computing and storing all T² entries is paying for rank that is not there.',
  mechanism: 'Project the *length* dimension of K and V from T down to a fixed k with a learned linear map, then run ordinary attention against those k keys.',
  buys: 'O(T·k) — linear in T — with the softmax left completely intact.',
  givesUp: 'The projection is over sequence positions, so T must be fixed at training time. That makes it unusable for autoregressive generation, where T grows by one every step. And "low-rank" is an empirical observation, not a guarantee.',
  pickWhen: 'Fixed-length encoder workloads — classification, retrieval. It never entered the LLM line for exactly the reason above, which is a good illustration that a mechanism can be right and still be wrong for your workload.',
  arxiv: '2006.04768', sourceLabel: 'Wang et al., arXiv:2006.04768',
  sourceUrl: 'https://arxiv.org/abs/2006.04768',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'linear-attention', date: '2020-06-29', name: 'Linear attention ("Transformers are RNNs")',
  short: 'Linear attention', pays: ['compute','memory'], required: true, anchor: true,
  problem: 'Softmax’s shared denominator ties every score to every other score — which is exactly what forces you to keep every individual old key around when a new query arrives. So: what happens if you drop it?',
  mechanism: 'Replace exp(q·k) with φ(q)·φ(k) for a non-negative feature map. Associativity then lets you regroup: instead of (QKᵀ)V, accumulate S = Σⱼ φ(kⱼ)vⱼᵀ once and read yᵢ = φ(qᵢ)S. The past collapses into one fixed-size matrix, and generation becomes a plain RNN update.',
  buys: 'O(T) training, and O(1) memory and O(1) time per generated token. The KV cache stops growing entirely. This is the only family on the timeline where context length costs literally nothing at inference.',
  givesUp: 'Everything softmax was doing: no competition between keys, no weights that sum to one, and — because the state is fixed-size — old memories interfere with new ones. Exact recall of one specific old token degrades badly, which is the well-documented weakness on retrieval and in-context tasks.',
  pickWhen: 'As the cheap majority of layers in a hybrid — never on its own if you need exact recall. Every production use on this timeline (MiniMax, Qwen3-Next, Kimi Linear) interleaves it with real attention layers.',
  arxiv: '2006.16236', sourceLabel: 'Katharopoulos et al., arXiv:2006.16236',
  sourceUrl: 'https://arxiv.org/abs/2006.16236',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'bigbird', date: '2020-07-28', name: 'BigBird (window + global + random)',
  short: 'BigBird', pays: ['compute'], required: false,
  problem: 'A sliding window alone has no long-range path within a single layer; purely random sparsity has no locality. Neither is enough by itself.',
  mechanism: 'Combine three patterns: a local window, a few global tokens, and a set of randomly chosen keys per query. The paper proves the result is still a universal approximator and Turing complete.',
  buys: 'Linear complexity with a theoretical guarantee that the sparse attention graph still connects everything in few hops. Strong long-document results.',
  givesUp: 'Random attention is hardware-hostile — gathering scattered keys destroys memory coalescing, so realized speedups lag well behind the FLOP count. And it is still content-independent: the random keys are random, not relevant.',
  pickWhen: 'Long-document encoders. Its lasting contribution is the argument that a sparse attention graph needs a short *diameter*, not merely few edges — which is why pure sliding window is not enough.',
  arxiv: '2007.14062', sourceLabel: 'Zaheer et al., arXiv:2007.14062',
  sourceUrl: 'https://arxiv.org/abs/2007.14062',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'performer', date: '2020-09-30', name: 'Performer (FAVOR+)',
  short: 'Performer', pays: ['compute','memory'], required: false,
  problem: 'Linear attention with an arbitrary feature map like elu+1 has no principled relationship to the softmax it is replacing. How wrong is it?',
  mechanism: 'FAVOR+ uses positive orthogonal random features so that φ(q)·φ(k) is an unbiased, low-variance estimator of exp(q·k) — approximating softmax attention itself rather than replacing it with something else.',
  buys: 'Linear time and space with provable, bounded approximation error to true softmax attention, and it can be retro-fitted to an already-trained transformer.',
  givesUp: 'It remains an estimator. Variance is real, and the approximation is worst precisely where softmax is sharp and peaked — which is exactly the retrieval-like case you most care about. In practice quality lagged on hard language tasks.',
  pickWhen: 'Largely superseded, but it is the strongest theoretical statement in the linear-attention family, and it clarified that the problem is not the linearity but what you lose when the distribution should be sharp.',
  arxiv: '2009.14794', sourceLabel: 'Choromanski et al., arXiv:2009.14794',
  sourceUrl: 'https://arxiv.org/abs/2009.14794',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'delta-rule', date: '2021-02-22', name: 'The delta rule in linear transformers',
  short: 'Delta rule', pays: ['quality'], required: true,
  problem: 'A linear-attention state that only accumulates (S ← S + vkᵀ) can never correct itself. Write "key A → 55" after "key A → 40" and a read of A returns 95, not 55. The state remembers, but it cannot revise.',
  mechanism: 'Before writing, read what the state currently returns for this key, subtract it, and write only the difference: S ← S + (v_new − S k)kᵀ. This is the classical Widrow–Hoff delta rule (1960), applied to the fast-weight matrix.',
  buys: 'The fixed-size state becomes an editable memory rather than an accumulator. Capacity is used far better, because superseded associations are removed instead of piling up on top of each other.',
  givesUp: 'In its natural form the update is sequential — each write depends on the state left by the previous write — so it forfeits the parallel-scan trick that made linear attention trainable at scale. This is why the idea sat almost unused for three years.',
  pickWhen: 'Conceptually mandatory for any fixed-size memory that must stay correct over time. Practically, use the 2024 parallelized formulation below.',
  arxiv: '2102.11174', sourceLabel: 'Schlag, Irie & Schmidhuber, arXiv:2102.11174',
  sourceUrl: 'https://arxiv.org/abs/2102.11174',
  dateBasis: 'arXiv API published field (v1). The delta rule itself is Widrow–Hoff 1960; this is the paper that brought it into linear attention, and it is the correct date for *this* mechanism.'
},
{
  id: 'rfa', date: '2021-03-03', name: 'Random Feature Attention (and gating)',
  short: 'Random Feature Attention', pays: ['compute','memory'], required: false,
  problem: 'Same question as Performer, plus a second one: a fixed-size state with no decay treats a token from 1M steps ago exactly like the previous token. Should it?',
  mechanism: 'Random Fourier features approximate the softmax kernel, and — the part that mattered most — a learned exponential decay is applied to the running state, so older contributions fade.',
  buys: 'A principled link back to softmax, and the introduction of *recency gating* to the linear family. That gating is the direct ancestor of the decay terms in Mamba, GLA and Gated DeltaNet.',
  givesUp: 'Still an estimator with real variance. And decay is a blunt instrument: it forgets by age, not by relevance, so a critical fact stated early is forgotten at the same rate as an irrelevant one.',
  pickWhen: 'Superseded as an architecture, but you are using its idea every time you use a gated linear model. Forgetting-by-age was the necessary step before forgetting-by-content.',
  arxiv: '2103.02143', sourceLabel: 'Peng et al., arXiv:2103.02143',
  sourceUrl: 'https://arxiv.org/abs/2103.02143',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'rope', date: '2021-04-20', name: 'Rotary Position Embedding (RoPE)',
  short: 'RoPE', pays: ['position'], required: true, anchor: true,
  problem: 'Absolute embeddings hit a length wall and encode index rather than distance. Shaw’s relative table clips at a maximum offset and is slow to fuse into a kernel. Neither is good enough.',
  mechanism: 'Treat each 2D pair of a head’s dimensions as an arrow in a plane and rotate it by m·θᵢ, where m is the position. Because a dot product depends only on the angle between two arrows, both absolute rotations cancel: ⟨R(m)q, R(n)k⟩ = ⟨q, R(n−m)k⟩. The positional part of the score depends only on n−m — the distance.',
  buys: 'Relative position for free. Applied to Q and K only, no extra parameters, no per-pair table, no length table to exhaust. It fuses cleanly into FlashAttention-style kernels, which is a large part of why it won. It is the default in essentially every open LLM since LLaMA.',
  givesUp: 'It is *defined* at any position but *trained* at none beyond the training length. High-frequency dimensions complete many full rotations inside the training window, so past it the model sees angle patterns it has never learned to read. Quality falls off a cliff unless you intervene — which is precisely why the next four entries exist.',
  pickWhen: 'The default choice today. Budget for a context-extension step alongside it from the start, rather than discovering you need one later.',
  arxiv: '2104.09864', sourceLabel: 'Su et al., RoFormer, arXiv:2104.09864',
  sourceUrl: 'https://arxiv.org/abs/2104.09864',
  dateBasis: 'arXiv API published field (v1). The v5 revision is 2023-11-08; RoPE is an April 2021 idea that took two years to become the default.'
},
{
  id: 'alibi', date: '2021-08-27', name: 'ALiBi (attention with linear biases)',
  short: 'ALiBi', pays: ['position'], required: true,
  problem: 'RoPE and sinusoidal both fail when asked to run far beyond their training length. Could position be handled so that extrapolation works by construction, rather than by later repair?',
  mechanism: 'Add no positional embedding at all. Instead subtract a linear penalty from every attention score in proportion to distance: score = q·k − m·|i−j|, with a different fixed slope m for each head so heads have different effective ranges.',
  buys: 'It genuinely extrapolates. Train at 1K, run at 2K and beyond with no degradation, no fine-tuning, no extra parameters and no runtime cost. It is faster and lighter than RoPE.',
  givesUp: 'The linear penalty is a hard recency prior: distant tokens are systematically suppressed regardless of how relevant they are. It does not so much *use* a long context as gracefully ignore the far part of it, and it underperforms on tasks that need real long-range retrieval.',
  pickWhen: 'When robustness to unexpected lengths matters more than deep long-range retrieval, or as a per-head bias inside a hybrid. BLOOM and MPT shipped it; the field mostly chose RoPE-plus-extension instead — a choice worth revisiting, not a settled verdict.',
  arxiv: '2108.12409', sourceLabel: 'Press, Smith & Lewis, arXiv:2108.12409',
  sourceUrl: 'https://arxiv.org/abs/2108.12409',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'flash-gau', date: '2022-02-21', name: 'Gated attention unit + chunked linear attention (FLASH)',
  short: 'Gated attention (FLASH)', pays: ['compute'], required: false,
  problem: 'Multi-head attention plus a wide FFN is two expensive blocks per layer. And linear-attention variants kept being unstable once made causal.',
  mechanism: 'A gated attention unit that merges the attention and FFN roles into a single block with one head, plus mixed chunk attention: quadratic attention *within* local chunks, linear attention *across* them.',
  buys: 'Large wall-clock speedups at equal quality, and it established local-quadratic / global-linear chunking as a workable structure.',
  givesUp: 'The single-head design sacrifices head diversity, results were sensitive to chunk size, and it never got the ecosystem support plain MHA has.',
  pickWhen: 'Its chunkwise formulation is now the standard way every linear and DeltaNet model is actually trained, so you use its ideas even without its architecture. It is also the origin of the "gated attention" now in Qwen3-Next and Step 3.5.',
  arxiv: '2202.10447', sourceLabel: 'Hua et al., arXiv:2202.10447',
  sourceUrl: 'https://arxiv.org/abs/2202.10447',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'flashattention', date: '2022-05-27', name: 'FlashAttention (IO-aware exact attention)',
  short: 'FlashAttention', pays: ['compute','memory','quality'], required: false, anchor: true,
  problem: 'Everyone assumed attention was compute-bound. It was not. Writing the T×T score matrix out to HBM and reading it back was the real cost — the arithmetic was almost incidental.',
  mechanism: 'Never materialize the score matrix at all. Tile Q, K and V into SRAM, compute softmax with a running max and running sum (online softmax), and accumulate the output tile by tile. Recompute in the backward pass instead of storing.',
  buys: 'Exact attention — bit-for-bit identical results, no approximation whatsoever — at 2–4× the speed and O(T) instead of O(T²) memory. This single result is why the field could keep using exact attention as contexts grew, and it deflated much of the 2019–2021 sparse-attention wave.',
  givesUp: 'Nothing about the mechanism; the cost is entirely engineering. It is a hand-written, hardware-specific kernel that must be rewritten for each GPU generation, and any custom per-pair attention bias (Shaw’s table, for instance) is hard to express inside it — which quietly pushed the whole field toward RoPE and ALiBi, both of which fuse trivially.',
  pickWhen: 'Always. This is not a trade-off, it is a strictly better implementation of something you were already doing. It belongs on this timeline because it changed which *mechanisms* were worth pursuing.',
  arxiv: '2205.14135', sourceLabel: 'Dao et al., arXiv:2205.14135',
  sourceUrl: 'https://arxiv.org/abs/2205.14135',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'h3', date: '2022-12-28', name: 'H3 — diagnosing why state models fail at language',
  short: 'H3 (SSM)', pays: ['compute','memory'], required: false,
  problem: 'State space models were far better than transformers on long-range synthetic benchmarks and much worse at actual language modelling. Nobody had a clean account of why.',
  mechanism: 'The paper diagnosed the gap as two specific missing abilities — recalling an earlier token, and comparing tokens across the sequence — then added a shift SSM plus a multiplicative interaction to supply them, trained with an FFT-based algorithm.',
  buys: 'The first SSM to come close to transformer perplexity on language while keeping O(T log T) training and O(1) per-step generation.',
  givesUp: 'It still needed a couple of genuine attention layers to match transformers — which is itself the finding. The FFT machinery is heavy and awkward on modern accelerators.',
  pickWhen: 'Superseded by Mamba, but H3’s diagnosis is the important artifact: "a fixed state fails at recall and comparison" is exactly the weakness that the delta rule and hybrid schedules spend the next four years attacking.',
  arxiv: '2212.14052', sourceLabel: 'Fu et al., arXiv:2212.14052',
  sourceUrl: 'https://arxiv.org/abs/2212.14052',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'gqa', date: '2023-05-22', name: 'Grouped-Query Attention (GQA)',
  short: 'GQA', pays: ['memory'], required: true, anchor: true,
  problem: 'MHA’s KV cache was too large; MQA’s was small but cost real quality and destabilized training. The only two options available were too far apart, and there was nothing in between.',
  mechanism: 'Partition the query heads into G groups and give each group one shared K/V head. G = h is MHA, G = 1 is MQA, and anything between interpolates. Crucially, an existing MHA checkpoint can be converted by mean-pooling its K/V heads and uptraining on about 5% of the original compute.',
  buys: 'Most of MQA’s cache saving with quality close to MHA — and you can convert models you have already trained rather than retraining from scratch. It is the default in Llama 2/3, Mistral, Qwen and almost everything since.',
  givesUp: 'It reduces the constant, not the growth. The cache is still O(T) per user: at 1M tokens a 4× smaller cache is still an enormous cache. And you do lose some head diversity.',
  pickWhen: 'Essentially always, as the baseline. The lecture’s verdict is exactly right — GQA is where you start, not where you finish, because it lowers the line without changing its slope.',
  arxiv: '2305.13245', sourceLabel: 'Ainslie et al., arXiv:2305.13245',
  sourceUrl: 'https://arxiv.org/abs/2305.13245',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'pi', date: '2023-06-27', name: 'Position Interpolation (PI)',
  short: 'Position Interpolation', pays: ['position'], required: false,
  problem: 'A RoPE model asked for position 4096 when it was trained to 2048 encounters rotation angles it has never seen, and the output degrades catastrophically — not gracefully.',
  mechanism: 'Do not extrapolate — interpolate. Divide every position index by the extension factor s, so position 4096 is presented to the model as 2048. Every angle stays inside the trained range. Then fine-tune briefly; 1000 steps was enough.',
  buys: 'Extends context 8–32× with a tiny fine-tune, and it is provably far more stable than extrapolation. It is also trivially simple to implement.',
  givesUp: 'Squeezing positions together compresses the high-frequency dimensions that distinguish *adjacent* tokens, so local resolution — "is this the previous word, or the one before it" — degrades. And it needs fine-tuning; there is no zero-shot version.',
  pickWhen: 'Superseded by NTK-aware and YaRN, which fix precisely this uniform-squashing problem. It stays on the timeline as the conceptual pivot: interpolate, do not extrapolate.',
  arxiv: '2306.15595', sourceLabel: 'Chen et al., arXiv:2306.15595',
  sourceUrl: 'https://arxiv.org/abs/2306.15595',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'ntk-aware', date: '2023-06-30', name: 'NTK-aware scaled RoPE',
  short: 'NTK-aware scaling', pays: ['position'], required: true, noPaper: true,
  problem: 'PI squashes every frequency equally, which blurs the high-frequency dimensions carrying local ordering. But those are exactly the dimensions that did not need help in the first place.',
  mechanism: 'Instead of scaling positions, change the RoPE base: 10000 → 10000·s^(d/(d−2)). Because dimension i’s frequency is base^(−2i/d), raising the base scales the low-frequency (long-wavelength) dimensions a great deal and the high-frequency ones almost not at all. Effectively: interpolate where the wavelength already exceeds the training window, extrapolate where it does not.',
  buys: 'Meaningful context extension with **zero fine-tuning** — which is the part that made it spread through the open-source community within days rather than months. Local resolution is preserved.',
  givesUp: 'It is a heuristic with no training-time guarantee. Some dimensions still end up out-of-distribution, and it degrades at large extension factors. It was also published as a Reddit post rather than a paper, so there is no controlled evaluation behind the original claim.',
  pickWhen: 'When you need a quick zero-shot extension and cannot fine-tune. If you can fine-tune, YaRN is strictly better.',
  arxiv: null, sourceLabel: 'u/bloc97, r/LocalLLaMA (no paper)',
  sourceUrl: 'https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/',
  dateBasis: 'THE WEAKEST DATE ON THIS PAGE, and flagged as such. There is no paper — the primary source is a Reddit post in late June 2023. The citable corroboration is YaRN (arXiv:2309.00071), which credits it as "bloc97, 2023" and reproduces the formula. Treat the day as approximate; the month and year are solid.'
},
{
  id: 'yarn', date: '2023-08-31', name: 'YaRN',
  short: 'YaRN', pays: ['position'], required: true,
  problem: 'PI blurs everything uniformly; NTK-aware is an untuned heuristic; and neither addresses what happens to attention *entropy* when you stretch the position axis — scores flatten, and the model attends to everything a bit.',
  mechanism: 'Three parts. NTK-by-parts: interpolate only the dimensions whose wavelength exceeds the training context and leave the rest untouched. A short fine-tune. And a temperature on the attention logits (t = 0.1·ln(s)+1) that corrects the entropy drift stretching introduces.',
  buys: 'State of the art RoPE extension: 64–128× context with roughly 10× fewer tokens and 2.5× fewer steps than PI, and it composes with a sliding-window trick for zero-shot use.',
  givesUp: 'Three interacting hyperparameters (α, β, and the temperature) that need per-model tuning, and it still wants fine-tuning for best results. Fundamentally it stretches a model into a regime it was not trained in — the long-context quality is real, but below native long training.',
  pickWhen: 'The default extension recipe when you have a good short-context model and a modest fine-tuning budget. Qwen, gpt-oss and many others ship it.',
  arxiv: '2309.00071', sourceLabel: 'Peng et al., arXiv:2309.00071',
  sourceUrl: 'https://arxiv.org/abs/2309.00071',
  dateBasis: 'arXiv API published field (v1). Its latest revision is dated 2026-02-06 — nearly two and a half years after v1, and a trap for anyone reading the top of the abstract page.'
},
{
  id: 'attention-sinks', date: '2023-09-29', name: 'Attention sinks (StreamingLLM)',
  short: 'Attention sinks', pays: ['memory','quality'], required: true,
  problem: 'Evict the oldest tokens from the KV cache to hold a fixed window, and perplexity explodes immediately. Not degrades — explodes. And nobody could say why.',
  mechanism: 'The diagnosis: softmax must sum to 1, so when a head has nothing relevant to attend to it has to dump its weight somewhere — and it learns to dump it on the first few tokens regardless of their content. Those are attention sinks. Evicting them removes the drain and throws every other score off. The fix is almost embarrassingly simple: always keep the first ~4 tokens, plus a rolling window.',
  buys: 'Stable generation over 4M+ tokens with a constant-size cache and a 22× speedup over recomputation — on models that are already trained, with no fine-tuning at all.',
  givesUp: 'This is streaming *stability*, not long-context *understanding*. The evicted middle is genuinely gone and the model cannot answer anything about it. It buys an infinite conversation, not an infinite memory — a distinction that marketing has repeatedly blurred.',
  pickWhen: 'Long-running chat or streaming where recent context is what matters. The deeper finding — that softmax needs an escape valve — is now designed in rather than discovered: gpt-oss ships a learned per-head sink logit.',
  arxiv: '2309.17453', sourceLabel: 'Xiao et al., arXiv:2309.17453',
  sourceUrl: 'https://arxiv.org/abs/2309.17453',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'mistral-swa', date: '2023-10-10', name: 'Sliding window in a production decoder (Mistral 7B)',
  short: 'SWA in production', pays: ['memory'], required: false,
  problem: 'Sliding window was well known from encoders, but nobody had shipped a strong open *decoder* built around it, so nobody knew whether it survived contact with generation.',
  mechanism: 'Every layer attends to a 4096-token window; stacking 32 layers gives a theoretical receptive field of about 131K. A rolling buffer cache of fixed size w means the cache never grows past the window.',
  buys: 'A constant-size KV cache regardless of conversation length, plus a genuine quality win at 7B. It made "the cache does not have to grow" a mainstream production idea rather than a research one.',
  givesUp: 'Long-range information must survive a game of telephone up through many layers, degrading at each hop, so the *usable* receptive field is far smaller than the theoretical one. Tellingly, Mistral quietly dropped SWA in later models.',
  pickWhen: 'As the cheap layer in an alternating schedule — which is exactly how Gemma 2/3, gpt-oss and Arcee Trinity use it now (roughly 5:1 local:global), rather than in every layer.',
  arxiv: '2310.06825', sourceLabel: 'Jiang et al., Mistral 7B, arXiv:2310.06825',
  sourceUrl: 'https://arxiv.org/abs/2310.06825',
  dateBasis: 'arXiv API published field (v1). The model was released via blog roughly two weeks earlier (2023-09-27); the paper date is used here for consistency with every other row, and the discrepancy is noted rather than hidden.'
},
{
  id: 'mamba', date: '2023-12-01', name: 'Mamba (selective state space)',
  short: 'Mamba', pays: ['compute','memory'], required: false,
  problem: 'SSMs were linear-time, but their dynamics were input-independent — the same recurrence applied to every token. So they could not selectively remember or forget, which is precisely why they failed at recall.',
  mechanism: 'Make the SSM parameters (Δ, B, C) functions of the input. That breaks the convolutional/FFT training trick, so it is replaced by a hardware-aware parallel scan that keeps the state in SRAM.',
  buys: 'Transformer-quality language modelling with O(T) training and O(1) per-token generation, 5× higher inference throughput, and quality that keeps improving with context out to 1M tokens.',
  givesUp: 'A fixed-size state still cannot do exact retrieval — pure Mamba underperforms on in-context learning and copying, the same wall H3 identified. It also needs custom kernels and has none of the tooling ecosystem attention enjoys.',
  pickWhen: 'Almost nobody ships pure Mamba for language; they ship hybrids (Jamba, Zamba, Nemotron-H). The lesson the field actually took away is "selectivity and gating are required", which is the G in Gated DeltaNet.',
  arxiv: '2312.00752', sourceLabel: 'Gu & Dao, arXiv:2312.00752',
  sourceUrl: 'https://arxiv.org/abs/2312.00752',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'griffin', date: '2024-02-29', name: 'Griffin — local attention + gated linear recurrence',
  short: 'Griffin (hybrid)', pays: ['compute','memory'], required: false, anchor: true,
  problem: 'A fixed-size recurrence cannot do exact local recall. Full attention is too expensive globally. So why is anyone choosing one mechanism for the entire network?',
  mechanism: 'Interleave gated linear recurrence blocks (RG-LRU) with local sliding-window attention blocks. The recurrence carries long-range compressed state; the local attention gives exact access to recent tokens.',
  buys: 'Matched Llama-2 quality on 6× less training data, faster inference than MQA, and extrapolation well beyond the training length.',
  givesUp: 'Global *exact* retrieval is still weak, because no layer anywhere sees the whole sequence exactly. Two block types also means two kernel paths and two cache types to serve — real operational complexity.',
  pickWhen: 'This is the architectural template the entire 2025–26 wave follows. The lecture’s D-D-D-G schedule is the same idea. Griffin is where "schedule across depth" became a mainstream design choice rather than an oddity.',
  arxiv: '2402.19427', sourceLabel: 'De et al., arXiv:2402.19427',
  sourceUrl: 'https://arxiv.org/abs/2402.19427',
  dateBasis: 'arXiv API published field (v1) — 29 Feb 2024, a leap day, which some citation tools silently normalize to 1 March.'
},
{
  id: 'infini-attention', date: '2024-04-10', name: 'Infini-attention (compressive memory)',
  short: 'Infini-attention', pays: ['memory'], required: false,
  problem: 'Sliding window throws the evicted past away entirely. StreamingLLM keeps it stable but still deletes the middle. Could the evicted past be *compressed* instead of discarded?',
  mechanism: 'In each attention layer, keep local sliding-window attention and add a compressive linear-attention memory alongside it. When a segment scrolls out of the window, its K/V are written into a fixed-size memory matrix using a delta-rule-style update rather than dropped. A learned gate blends the local read and the memory read.',
  buys: 'Unbounded context in bounded memory — a 114× compression ratio, and 1M-token passkey retrieval from a 1B model, with a constant memory footprint.',
  givesUp: 'Everything outside the window is now a lossy summary, so multi-fact reasoning over the far past degrades even when single-needle retrieval still works. Independent reproduction proved difficult, and adoption was thin.',
  pickWhen: 'Worth knowing as the clearest statement of "compress the evicted past instead of deleting it". In practice the field went the other way — sparse retrieval over a fully stored cache — which is a useful reminder that the elegant option does not always win.',
  arxiv: '2404.07143', sourceLabel: 'Munkhdalai, Faruqui & Gopal, arXiv:2404.07143',
  sourceUrl: 'https://arxiv.org/abs/2404.07143',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'mla', date: '2024-05-07', name: 'Multi-head Latent Attention (MLA)',
  short: 'MLA', pays: ['memory'], required: true, anchor: true,
  problem: 'GQA saves cache by making heads share a K/V, which costs head diversity. Is there a way to shrink the cache without forcing the heads to agree on one view of the past?',
  mechanism: 'Compress K and V jointly into a single low-rank latent vector c, and cache only c. At attention time, up-project back to full per-head K and V — and because that up-projection is linear, it can be absorbed into the query and output weight matrices, so it costs nothing extra at inference. RoPE cannot be absorbed this way, so a small decoupled subset of dimensions carries RoPE separately.',
  buys: '93.3% KV cache reduction versus DeepSeek-67B, with quality reported *better* than MHA rather than merely close. It keeps full per-head expressiveness — the exact thing GQA cannot do.',
  givesUp: 'Substantially more complex, with an awkward decoupled-RoPE carve-out that every implementation has to special-case. Peak activation memory during up-projection is higher, converting an existing GQA checkpoint is not trivial, and kernel support took a long time to arrive.',
  pickWhen: 'Large models where serving memory dominates and you control the training run. It has now spread well beyond DeepSeek — GLM-5, Sarvam 105B and Ling 2.5 all adopt it.',
  arxiv: '2405.04434', sourceLabel: 'DeepSeek-AI, DeepSeek-V2, arXiv:2405.04434',
  sourceUrl: 'https://arxiv.org/abs/2405.04434',
  dateBasis: 'arXiv API published field (v1). MLA was introduced in the DeepSeek-V2 report — there is no standalone MLA paper, so V2 is the correct primary source.'
},
{
  id: 'deltanet-parallel', date: '2024-06-10', name: 'DeltaNet parallelized over sequence length',
  short: 'DeltaNet (parallel)', pays: ['quality','compute'], required: true,
  problem: 'The delta rule made a fixed state editable, but its update was sequential — so it could not be trained at scale. That single engineering fact kept a good idea out of real models for three years.',
  mechanism: 'Recognize the delta update as a generalized Householder transform and reparameterize it via the WY representation. That turns the chunk-level recurrence into matrix multiplications, so it trains with a parallel chunkwise algorithm.',
  buys: 'The delta rule finally becomes trainable at scale. DeltaNet beats Mamba and GLA on associative recall and language modelling — exactly where fixed-state models had been weakest.',
  givesUp: 'It has no forgetting mechanism. The state edits but never decays, so over very long sequences irrelevant old associations linger and clutter a fixed capacity. It also underperformed on tasks where a decaying model was better, which is what motivated the gated version six months later.',
  pickWhen: 'Superseded by Gated DeltaNet, which is this plus decay. Its importance is as the enabling engineering result — a reminder that "trainable in parallel" is a first-class design constraint, not an implementation detail.',
  arxiv: '2406.06484', sourceLabel: 'Yang, Wang, Zhang & Kim, arXiv:2406.06484',
  sourceUrl: 'https://arxiv.org/abs/2406.06484',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'quest', date: '2024-06-16', name: 'Quest — query-aware top-k page selection',
  short: 'Quest (top-k)', pays: ['compute'], required: true,
  problem: 'You cannot pick the top-k keys without first scoring all T keys — which is precisely the cost you were trying to avoid. Naive top-k saves the value work and none of the scoring work.',
  mechanism: 'Split the KV cache into pages and store per-page per-channel min/max bounds. For a query, compute an *upper bound* on the attention score achievable within each page from those bounds — an operation whose cost scales with page count, not token count — then run exact attention only on the top-scoring pages.',
  buys: '7.03× self-attention speedup and 2.23× end-to-end latency reduction with negligible accuracy loss, applied to models you already have, with no retraining.',
  givesUp: 'It is inference-only: the model was trained dense, so the sparsity is a post-hoc approximation, and the upper bound is loose enough that a genuinely needed page can be missed. It saves compute but not memory — the entire cache must still be stored.',
  pickWhen: 'Serving an existing long-context model where decode latency hurts. It also names the exact problem — cheap candidate proposal — that NSA and DSA go on to solve by training the selector into the model.',
  arxiv: '2406.10774', sourceLabel: 'Tang et al., arXiv:2406.10774',
  sourceUrl: 'https://arxiv.org/abs/2406.10774',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'gated-deltanet', date: '2024-12-09', name: 'Gated DeltaNet',
  short: 'Gated DeltaNet', pays: ['quality','memory'], required: true, anchor: true,
  problem: 'Mamba2 has decay but no targeted editing. DeltaNet has targeted editing but no decay. Each one fails exactly where the other succeeds, which is a strong hint they belong together.',
  mechanism: 'Combine them: Sₜ = αₜ(Sₜ₋₁ − βₜSₜ₋₁kkᵀ) + βₜvkᵀ. The gate α erases globally when the context shifts topic; the delta term edits precisely for one key. Trained with a chunkwise parallel algorithm inherited from the DeltaNet work.',
  buys: 'Beats Mamba2 and DeltaNet on language modelling, common-sense reasoning, long-context and — most importantly — recall-intensive tasks. This is currently the strongest linear-attention block available.',
  givesUp: 'It is still a fixed-size state. It narrows the recall gap to attention without closing it, which is precisely why every single deployment of it is a hybrid rather than a replacement. Two interacting gates also add tuning burden and kernel complexity.',
  pickWhen: 'As the D layers of a hybrid. Qwen3-Next, Qwen3.5 and Qwen3-Coder-Next all run it at 3:1 against full attention; Kimi Linear runs its own variant at the same ratio.',
  arxiv: '2412.06464', sourceLabel: 'Yang, Kautz & Hatamizadeh, arXiv:2412.06464',
  sourceUrl: 'https://arxiv.org/abs/2412.06464',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'lightning-minimax', date: '2025-01-14', name: 'Lightning Attention at frontier scale (MiniMax-01)',
  short: 'Lightning Attention', pays: ['compute','memory'], required: false,
  problem: 'Linear attention had never been demonstrated at frontier scale. Did the quality gap actually close, or did it merely look acceptable at 1B parameters and quietly reopen at 400B?',
  mechanism: 'Lightning Attention — an IO-aware tiled linear-attention kernel with a decay term — deployed inside a 456B MoE at a 7:1 ratio (seven lightning layers per one softmax layer), trained to 1M context and served at 4M.',
  buys: 'The existence proof. A 4M-token context at frontier quality with near-linear compute scaling. It moved hybrid linear attention from research curiosity to production reality.',
  givesUp: 'The 7:1 ratio was chosen, not ablated — the same criticism the lecture levels at DDDGDDDG. And the softmax layers it retains still carry an O(T) cache, so the memory bill is reduced by a constant factor, not removed.',
  pickWhen: 'Its value here is as evidence rather than as a recipe: at 456B, a mostly-linear model is competitive. That result is what licensed the whole 2025–26 hybrid wave.',
  arxiv: '2501.08313', sourceLabel: 'MiniMax et al., arXiv:2501.08313',
  sourceUrl: 'https://arxiv.org/abs/2501.08313',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'nsa', date: '2025-02-16', name: 'Native Sparse Attention (NSA)',
  short: 'NSA', pays: ['compute'], required: true, anchor: true,
  problem: 'Sparse attention kept being an inference-time patch applied to a densely-trained model. That mismatch caps quality — the model never learned to work with the sparsity. And most sparse patterns were too irregular to actually be fast on a GPU anyway.',
  mechanism: 'Three branches per query, combined by a learned gate: compressed coarse-grained blocks for global context, selected fine-grained blocks chosen using scores derived from the compression branch, and a sliding window for locality. The block structure is chosen so it maps onto Tensor Core memory access patterns.',
  buys: 'Trained sparse from scratch, so the model learns to use the sparsity — it matched or beat full attention rather than approximating it. 9× forward and 6× backward speedup at 64K, 11.6× decode. Crucially, the compression branch *supplies* the selection scores, which is how it avoids scoring every key.',
  givesUp: 'It must be trained this way; you cannot bolt it onto an existing dense model, which rules out every checkpoint you already own. Three branches plus custom kernels is substantial complexity, and block granularity means selection is coarse.',
  pickWhen: 'New long-context pretraining runs where you control the architecture. This is the paper that made "natively trainable sparse attention" the standard framing.',
  arxiv: '2502.11089', sourceLabel: 'Yuan et al. (DeepSeek), arXiv:2502.11089',
  sourceUrl: 'https://arxiv.org/abs/2502.11089',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'moba', date: '2025-02-18', name: 'MoBA (Mixture of Block Attention)',
  short: 'MoBA', pays: ['compute'], required: false,
  problem: 'The same goal as NSA, arrived at two days later by a different route: how do you let the model choose which blocks to read, without hand-designing the branches?',
  mechanism: 'Partition the KV into blocks, mean-pool each block into one representative key, score the query against those representatives, and apply a top-k gate. It is Mixture-of-Experts, except the experts are blocks of context. It can also switch back to full attention at will.',
  buys: '6.5× speedup at 1M tokens and 16× at 10M — and because it can toggle back to full attention, it drops into an existing model with minimal loss. Deployed in Kimi.',
  givesUp: 'Mean-pooling a block is a crude summary: one sharp key inside an otherwise dull block gets averaged into invisibility. Top-k is non-differentiable, so gate training is indirect, and block granularity limits precision.',
  pickWhen: 'When you want trainable sparsity but need to stay compatible with an existing dense model. NSA and MoBA landing 48 hours apart is the single clearest signal of where the field’s attention was in early 2025 — and the timeline is the only view that makes that visible.',
  arxiv: '2502.13189', sourceLabel: 'Lu et al. (Moonshot AI), arXiv:2502.13189',
  sourceUrl: 'https://arxiv.org/abs/2502.13189',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'log-linear', date: '2025-06-05', name: 'Log-Linear Attention',
  short: 'Log-Linear Attention', pays: ['memory','quality'], required: false,
  problem: 'The choice looked binary: O(T) with a fixed state that forgets, or O(T²) with perfect recall. There was nothing in between, and the field had stopped looking for anything.',
  mechanism: 'Replace the single fixed-size state with a set of states over exponentially growing time spans — a Fenwick-tree partition of the past. Recent history gets fine-grained states; distant history gets progressively coarser ones.',
  buys: 'O(log T) state and O(T log T) training, giving a hierarchical memory: near-exact on recent tokens, progressively summarized further back. It applies to Mamba-2 and Gated DeltaNet alike rather than being a new architecture.',
  givesUp: 'A log-growing state is not a constant state, so the O(1) inference promise of linear attention is gone. The hierarchy is a fixed structure imposed on the past rather than learned, and kernel efficiency lags the theory.',
  pickWhen: 'Not yet in production. It is on this timeline because it is the most interesting answer to "what comes after the fixed state", and because a third option between linear and quadratic is exactly the kind of move the timeline suggests should come next.',
  arxiv: '2506.04761', sourceLabel: 'Guo et al., arXiv:2506.04761',
  sourceUrl: 'https://arxiv.org/abs/2506.04761',
  dateBasis: 'arXiv API published field (v1). Published at ICLR 2026 — citing the conference year would misdate it by roughly seven months.'
},
{
  id: 'gpt-oss-sinks', date: '2025-08-08', name: 'Learned attention sinks + alternating SWA (gpt-oss)',
  short: 'Learned sinks', pays: ['memory','quality'], required: false,
  problem: 'StreamingLLM showed that models spontaneously create attention sinks on whatever token happens to be first. If a sink is structurally necessary, why leave it to chance on a real token that also has to carry meaning?',
  mechanism: 'Alternating banded (128-token) and dense attention layers, GQA with 8 KV heads, YaRN out to 131K, and a learned per-head scalar appended to the softmax denominator — a dedicated sink that attends to nothing at all.',
  buys: 'The escape valve becomes explicit and content-free, so no real token has its representation distorted by carrying sink duty. That in turn makes sliding-window layers safe to use aggressively.',
  givesUp: 'Alternating layer types complicates serving. The 128-token band is very narrow, so most layers see almost nothing globally and the model leans heavily on the sparse dense layers.',
  pickWhen: 'Cited here because it is the point where the 2023 StreamingLLM *diagnosis* became a designed-in architectural *component*. Gemma and others do something similar. It is a good example of the field converting a bug report into a feature.',
  arxiv: '2508.10925', sourceLabel: 'OpenAI, gpt-oss model card, arXiv:2508.10925',
  sourceUrl: 'https://arxiv.org/abs/2508.10925',
  dateBasis: 'arXiv API published field (v1) for the model card. The weights were released three days earlier, on 2025-08-05.'
},
{
  id: 'qwen3-next', date: '2025-09-11', name: 'Gated DeltaNet 3:1 hybrid in production (Qwen3-Next)',
  short: 'Qwen3-Next 3:1', pays: ['memory','compute'], required: false, noPaper: true,
  problem: 'Gated DeltaNet was the strongest linear block on benchmarks. Would that survive inside a large production MoE serving real traffic?',
  mechanism: 'A 3:1 depth schedule — three Gated DeltaNet layers to one gated full-attention layer — inside an 80B MoE with 3B active parameters, plus multi-token prediction.',
  buys: 'About 10× the inference throughput of Qwen3-32B at long context, with quality held. This is the moment the D-D-D-G motif became a shipped default rather than a research finding.',
  givesUp: 'The 3:1 ratio is inherited rather than ablated — precisely the criticism the lecture makes of DDDGDDDG. And two layer types means two kernel paths and two cache types to serve.',
  pickWhen: 'It is the reference point for anyone choosing a hybrid ratio today. Qwen3.5 and Qwen3-Coder-Next both kept the same layout into 2026, which is either accumulating evidence or accumulating inertia — nobody has run the ablation to tell you which.',
  arxiv: null, sourceLabel: 'vLLM day-0 support blog, 2025-09-11',
  sourceUrl: 'https://blog.vllm.ai/2025/09/11/qwen3-next.html',
  dateBasis: 'No arXiv paper for Qwen3-Next specifically. Dated from the vLLM day-0 support post (2025-09-11), which coincides with Qwen’s own announcement week. The Qwen3 technical report (arXiv:2505.09388, 2025-05-14) is a different, earlier model and must not be used as this date.'
},
{
  id: 'dsa', date: '2025-09-29', name: 'DeepSeek Sparse Attention (DSA)',
  short: 'DSA', pays: ['compute'], required: true,
  problem: 'NSA’s selector was tied to its own compression branch. Could a smaller, cheaper, standalone indexer do the selecting instead — and could it be retrofitted onto an MLA model that already exists?',
  mechanism: 'A "lightning indexer" — a few heads running at low precision (FP8) — scores which past tokens matter. Fine-grained token-level top-k (2048 tokens) attention then runs only on those. Built on top of MLA, and trained in via continued pretraining from V3.1-Terminus rather than from scratch.',
  buys: 'Fine-grained token-level rather than block-level sparsity, with API price cuts of over 50% at parity quality. The structurally important idea is that the indexer is *separate and cheap* — deciding where to look is now its own component.',
  givesUp: 'The indexer is an extra component that must itself be trained and can rank wrongly, with no bound on how wrong. The full KV cache is still stored, so this saves compute but not memory. DeepSeek shipped it explicitly labelled experimental.',
  pickWhen: 'It is the direct predecessor of DeepSeek-V4’s CSA — the step where the field cleanly separated "decide where to look" from "look".',
  arxiv: null, sourceLabel: 'DeepSeek-V3.2-Exp release, 2025-09-29',
  sourceUrl: 'https://api-docs.deepseek.com/news/news250929',
  dateBasis: 'Official DeepSeek API news post dated 2025-09-29, plus the V3.2-Exp technical report in the GitHub release. There is no arXiv paper for DSA itself, so the release post is the primary source.'
},
{
  id: 'kimi-linear', date: '2025-10-30', name: 'Kimi Delta Attention (Kimi Linear)',
  short: 'Kimi Linear (KDA)', pays: ['memory','compute'], required: false,
  problem: 'Hybrids kept RoPE in their full-attention layers out of habit. But if the linear layers already encode position implicitly through their recurrence, is RoPE still doing anything there?',
  mechanism: 'Kimi Delta Attention — Gated DeltaNet with a fine-grained *per-channel* diagonal gate instead of a single scalar — at a 3:1 ratio against full-attention layers that use NoPE, no positional encoding at all. The KDA layers are left to carry position for the whole model.',
  buys: '75% KV cache reduction and up to 6× decoding throughput at 1M context, while *beating* full attention on quality. It is the first hybrid to claim it wins rather than merely holds parity.',
  givesUp: 'Per-channel gating costs more state and more kernel complexity than a scalar gate. And NoPE in the attention layers makes those layers wholly dependent on the KDA layers for position — an elegant coupling, but an untested one at extreme lengths.',
  pickWhen: 'Currently the strongest published argument that a hybrid can beat, not merely approximate, full attention. If that result holds under replication it changes the default.',
  arxiv: '2510.26692', sourceLabel: 'Kimi Team, arXiv:2510.26692',
  sourceUrl: 'https://arxiv.org/abs/2510.26692',
  dateBasis: 'arXiv API published field (v1).'
},
{
  id: 'drope', date: '2025-12-13', name: 'DroPE — drop the positional embeddings, then recalibrate',
  short: 'DroPE', pays: ['position'], required: true, anchor: true, flagged: true,
  problem: 'Every RoPE extension method — PI, NTK-aware, YaRN — is a repair applied to a positional scheme the model has already overfitted to. What if the positional embedding itself is the thing blocking length generalization?',
  mechanism: 'Remove RoPE from every layer of an already-pretrained model, then recalibrate for a short run — under 1% of the original pretraining budget (reported at 30–120B tokens). The claim is that explicit positional embeddings are a *training scaffold* needed for convergence, not a permanent architectural requirement: once the model is trained, causal masking alone supplies enough positional signal.',
  buys: 'Zero-shot context extension that preserves in-context performance and outperforms RoPE-scaling methods on RULER and LongBench, at under 1% of pretraining cost. No α/β/temperature hyperparameter hunt.',
  givesUp: 'It is a training procedure, not an inference switch — you need a recalibration budget and the ability to continue pretraining, which rules it out if you only have weights. Removing position entirely means leaning on an implicit signal whose behaviour at extreme lengths is still being established.',
  pickWhen: 'When you have a strong short-context model plus compute for a small continued-pretraining run, and you want extension without YaRN’s tuning burden. This is the method behind the lecture’s 8K → 256K, 32× claim.',
  arxiv: '2512.12167', sourceLabel: 'Gelberg, Eguchi, Akiba & Cetin (Sakana AI), arXiv:2512.12167',
  sourceUrl: 'https://arxiv.org/abs/2512.12167',
  dateBasis: 'arXiv API published field (v1), 13 Dec 2025. ⚠ DO NOT CONFUSE with arXiv:2503.15029 "DRoPE: Directional Rotary Position Embedding", a March 2025 autonomous-driving trajectory paper — that is the first hit for "DroPE arXiv" and it is a different thing entirely. Code: github.com/SakanaAI/DroPE'
},
{
  id: 'hybrid-2026', date: '2026-02-15', name: 'Hybrid attention goes mainstream — and the field splits',
  short: 'The 2026 split', pays: ['memory','compute'], required: false, noPaper: true,
  problem: 'By early 2026 the question was no longer whether hybrids work. It was which family — linear state, or sequence compression — should be the primary long-context lever.',
  mechanism: 'A cluster of frontier releases inside three weeks: Qwen3.5 (Gated DeltaNet hybrid), Qwen3-Coder-Next (3:1 Gated DeltaNet + gated attention), GLM-5 (MLA + sparse attention), Ling 2.5 1T (Lightning Attention + MLA), and MiniMax M2.5 going back to plain GQA.',
  buys: 'The genuinely useful signal here is the disagreement. In a single month, at frontier scale, four labs picked four different attention stacks and all shipped competitive models. That is evidence the choice is workload-dependent rather than solved.',
  givesUp: 'No controlled comparison exists between any of them — different data, different scales, different budgets. Nobody can currently tell you which family wins at matched budget, which is exactly the open question the lecture ends on.',
  pickWhen: 'Treat this row as the field’s current state of uncertainty, not as a recommendation. It is the honest answer to "what should I use in 2026": it depends, and anyone who tells you otherwise is guessing.',
  arxiv: null, sourceLabel: 'Raschka, "A Dream of Spring for Open-Weight LLMs" (Jan–Feb 2026 survey)',
  sourceUrl: 'https://magazine.sebastianraschka.com/p/a-dream-of-spring-for-open-weight',
  dateBasis: 'Dated from the individual model releases surveyed there: Qwen3.5 2026-02-15, Ling 2.5 2026-02-16, GLM-5 and MiniMax M2.5 2026-02-12, Qwen3-Coder-Next 2026-02-03. This row is a cluster, not one launch, and is labelled as such rather than pretending it is a single dated event.'
},
{
  id: 'deepseek-v4', date: '2026-04-26', name: 'Compressed Sparse Attention + Heavily Compressed Attention (DeepSeek-V4)',
  short: 'CSA + HCA', pays: ['memory','compute'], required: true, anchor: true,
  problem: 'MLA shrinks the cache per token and DSA cuts the compute per query — but a 1M-token context still stores an entry for every one of those million tokens. Nobody had attacked the number of stored *positions*.',
  mechanism: 'Two mechanisms in one hybrid. Compressed Sparse Attention merges small groups of tokens (typically 4) into one KV entry using a learned, data-dependent, per-dimension compressor, then applies DSA-style top-k over those compressed entries. Heavily Compressed Attention uses a far larger compression rate but keeps dense attention over the few entries that remain. Plus low-rank query and output projections.',
  buys: 'KV cache down to roughly 2% of a standard transformer at 1M context. Versus V3.2: 27% of the single-token inference FLOPs and 10% of the cache. It attacks stored positions and read positions at the same time, which nothing before it did together.',
  givesUp: 'Compression is lossy by construction — one entry now speaks for four tokens, and token-level detail in the far past is simply gone. Two attention types plus an indexer plus low-rank projections is a great deal of machinery, and it must be trained natively this way rather than retrofitted.',
  pickWhen: 'The current frontier answer for million-token contexts. Note that it takes the lecture’s "Road 2" — build long and train long — rather than extending a short model, which is the opposite bet from DroPE.',
  arxiv: '2606.19348', sourceLabel: 'DeepSeek-AI, arXiv:2606.19348',
  sourceUrl: 'https://arxiv.org/abs/2606.19348',
  dateBasis: 'arXiv API published field (v1), 26 Apr 2026.'
}
];

/* --- derived helpers, all computed from the array above ------------------- */
MECHANISMS.sort((a, b) => a.date.localeCompare(b.date) || a.name.localeCompare(b.name));

const YEAR_MIN = 2014, YEAR_MAX = 2027;
function toFrac(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  const y = d.getUTCFullYear();
  const start = Date.UTC(y, 0, 1), end = Date.UTC(y + 1, 0, 1);
  return y + (d.getTime() - start) / (end - start);
}
function pct(iso) {
  return ((toFrac(iso) - YEAR_MIN) / (YEAR_MAX - YEAR_MIN)) * 100;
}
const FMT = { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' };
function prettyDate(iso) {
  return new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', FMT);
}
