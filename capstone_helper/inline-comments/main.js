'use strict';

const obsidian = require('obsidian');
const { Plugin, ItemView, Notice, PluginSettingTab, Setting, MarkdownView, Menu, setIcon, debounce } = obsidian;
const { StateField, StateEffect } = require('@codemirror/state');
const { Decoration, EditorView, ViewPlugin } = require('@codemirror/view');

const VIEW_TYPE = 'inline-comments-view';
const CONTEXT = 40;

/* =====================================================================
 * Helpers
 * ===================================================================== */

function uid() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

function truncate(s, n) {
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

function relTime(ts) {
  const diff = Date.now() - ts;
  const m = Math.floor(diff / 60000);
  if (m < 1) return 'just now';
  if (m < 60) return m + 'm ago';
  const h = Math.floor(m / 60);
  if (h < 24) return h + 'h ago';
  const d = Math.floor(h / 24);
  if (d < 7) return d + 'd ago';
  return new Date(ts).toLocaleDateString();
}

function commonSuffixLen(a, b) {
  let i = 0;
  while (i < a.length && i < b.length && a[a.length - 1 - i] === b[b.length - 1 - i]) i++;
  return i;
}

function commonPrefixLen(a, b) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return i;
}

/** Find the best place in `text` for a comment's quoted text, using prefix/suffix context. */
function findAnchor(text, c) {
  if (!c.quote) return null;
  let best = null;
  let bestScore = -1;
  let bestDist = Infinity;
  let idx = text.indexOf(c.quote);
  let count = 0;
  while (idx !== -1 && count < 500) {
    const before = text.slice(Math.max(0, idx - (c.prefix || '').length), idx);
    const after = text.slice(idx + c.quote.length, idx + c.quote.length + (c.suffix || '').length);
    const score = commonSuffixLen(before, c.prefix || '') + commonPrefixLen(after, c.suffix || '');
    const dist = Math.abs(idx - c.from);
    if (score > bestScore || (score === bestScore && dist < bestDist)) {
      best = idx;
      bestScore = score;
      bestDist = dist;
    }
    idx = text.indexOf(c.quote, idx + 1);
    count++;
  }
  return best;
}

/* =====================================================================
 * CodeMirror 6 state: highlights for comments in the editor
 * ===================================================================== */

const setItems = StateEffect.define();
const setActiveEffect = StateEffect.define();
const setPendingEffect = StateEffect.define();

function buildDeco(value) {
  const ranges = [];
  for (const it of value.items) {
    if (it.orphaned || !it.visible || it.to <= it.from) continue;
    ranges.push(
      Decoration.mark({
        class: 'obc-hl' + (it.id === value.active ? ' obc-active' : ''),
        attributes: { 'data-obc-id': it.id },
      }).range(it.from, it.to)
    );
  }
  if (value.pending && value.pending.to > value.pending.from) {
    ranges.push(Decoration.mark({ class: 'obc-pending' }).range(value.pending.from, value.pending.to));
  }
  return Decoration.set(ranges, true);
}

const commentField = StateField.define({
  create() {
    return { items: [], active: null, pending: null, deco: Decoration.none };
  },
  update(value, tr) {
    let { items, active, pending } = value;
    let changed = false;

    if (tr.docChanged) {
      items = items.map((it) => {
        if (it.orphaned) return it;
        const from = tr.changes.mapPos(it.from, 1);
        const to = tr.changes.mapPos(it.to, -1);
        if (to <= from) return Object.assign({}, it, { from, to: from, orphaned: true });
        return Object.assign({}, it, { from, to });
      });
      if (pending) {
        const pf = tr.changes.mapPos(pending.from, 1);
        const pt = tr.changes.mapPos(pending.to, -1);
        pending = pt > pf ? { from: pf, to: pt } : null;
      }
      changed = true;
    }

    for (const e of tr.effects) {
      if (e.is(setItems)) {
        items = e.value;
        changed = true;
      } else if (e.is(setActiveEffect)) {
        active = e.value;
        changed = true;
      } else if (e.is(setPendingEffect)) {
        pending = e.value;
        changed = true;
      }
    }

    if (!changed) return value;
    const next = { items, active, pending };
    next.deco = buildDeco(next);
    return next;
  },
  provide: (f) => EditorView.decorations.from(f, (v) => v.deco),
});

function interactions(plugin) {
  return [
    EditorView.domEventHandlers({
      click(e) {
        const t = e.target;
        const el = t instanceof HTMLElement ? t.closest('.obc-hl') : null;
        if (el) plugin.onHighlightClick(el.getAttribute('data-obc-id'));
        return false;
      },
    }),
    ViewPlugin.fromClass(
      class {
        update(u) {
          if (u.docChanged) plugin.onDocChanged(u.view);
        }
      }
    ),
  ];
}

/* =====================================================================
 * Plugin
 * ===================================================================== */

class InlineCommentsPlugin extends Plugin {
  async onload() {
    await this.loadState();
    this.commentMode = false;
    this.activeId = null;
    this.lastPath = null;
    this.composer = null;
    this.drafts = new Map();

    this.requestSave = debounce(() => this.saveState(), 600, true);
    this.refreshViewsSoon = debounce(() => this.refreshViews(), 300, true);
    this.onLayoutSoon = debounce(() => this.onLayout(), 50, true);

    this.applyColor();

    this.registerView(VIEW_TYPE, (leaf) => new CommentsView(leaf, this));
    this.registerEditorExtension([commentField, interactions(this)]);

    // Ribbon + status bar
    this.ribbonEl = this.addRibbonIcon('message-square-plus', 'Toggle comment mode', () =>
      this.setCommentMode(!this.commentMode)
    );
    this.statusEl = this.addStatusBarItem();
    this.statusEl.addClass('obc-status');
    this.statusEl.addEventListener('click', () => this.setCommentMode(!this.commentMode));
    this.updateStatus();

    // Commands
    this.addCommand({
      id: 'toggle-comment-mode',
      name: 'Toggle comment mode',
      callback: () => this.setCommentMode(!this.commentMode),
    });
    this.addCommand({
      id: 'add-comment',
      name: 'Add comment to selection',
      checkCallback: (checking) => {
        const view = this.getEditorView();
        if (!view) return false;
        const sel = view.editor.cm.state.selection.main;
        if (sel.empty) return false;
        if (!checking) this.openComposer(view, sel.from, sel.to);
        return true;
      },
    });
    this.addCommand({
      id: 'open-comments-sidebar',
      name: 'Open comments sidebar',
      callback: () => this.activateView(),
    });

    // Right-click menu
    this.registerEvent(
      this.app.workspace.on('editor-menu', (menu, editor) => {
        if (!editor.getSelection().trim()) return;
        menu.addItem((item) =>
          item
            .setTitle('Add comment')
            .setIcon('message-square-plus')
            .onClick(() => {
              const view = this.getEditorView();
              if (!view) return;
              const sel = view.editor.cm.state.selection.main;
              this.openComposer(view, sel.from, sel.to);
            })
        );
      })
    );

    // Workspace / vault events
    this.registerEvent(this.app.workspace.on('file-open', () => this.onLayoutSoon()));
    this.registerEvent(this.app.workspace.on('active-leaf-change', () => this.onLayoutSoon()));
    this.registerEvent(this.app.workspace.on('layout-change', () => this.onLayoutSoon()));
    this.registerEvent(
      this.app.vault.on('rename', (file, oldPath) => {
        if (this.comments[oldPath]) {
          this.comments[file.path] = this.comments[oldPath];
          delete this.comments[oldPath];
          if (this.lastPath === oldPath) this.lastPath = file.path;
          this.requestSave();
          this.refreshViewsSoon();
        }
      })
    );
    this.registerEvent(
      this.app.vault.on('delete', (file) => {
        if (this.comments[file.path]) {
          delete this.comments[file.path];
          this.requestSave();
          this.refreshViewsSoon();
        }
      })
    );

    this.setupSelectionUI();
    this.addSettingTab(new InlineCommentsSettingTab(this.app, this));

    this.app.workspace.onLayoutReady(() => this.onLayout());
  }

  onunload() {
    document.body.classList.remove('obc-comment-mode');
    document.body.style.removeProperty('--obc-color');
    this.closeComposer();
    this.saveState();
  }

  /* ---------- persistence ---------- */

  async loadState() {
    const d = (await this.loadData()) || {};
    this.settings = Object.assign({ author: 'Me', showResolved: false, color: '#f5c518' }, d.settings);
    this.comments = d.comments || {};
  }

  async saveState() {
    await this.saveData({ settings: this.settings, comments: this.comments });
  }

  applyColor() {
    document.body.style.setProperty('--obc-color', this.settings.color);
  }

  /* ---------- comment mode ---------- */

  setCommentMode(on) {
    this.commentMode = on;
    document.body.classList.toggle('obc-comment-mode', on);
    if (this.ribbonEl) this.ribbonEl.toggleClass('is-active', on);
    this.updateStatus();
    this.hideFab();
    this.refreshViews();
    new Notice(on ? 'Comment mode on — select text to comment' : 'Comment mode off');
    if (on) this.activateView();
  }

  updateStatus() {
    this.statusEl.setText(this.commentMode ? '💬 Comment mode: on' : '💬 Comment mode: off');
    this.statusEl.toggleClass('is-on', this.commentMode);
  }

  /* ---------- editors <-> data ---------- */

  getEditorView() {
    const v = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (!v || !v.file || v.getMode() !== 'source' || !v.editor || !v.editor.cm) return null;
    return v;
  }

  markdownViews() {
    return this.app.workspace
      .getLeavesOfType('markdown')
      .map((l) => l.view)
      .filter((v) => v instanceof MarkdownView);
  }

  onLayout() {
    const active = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (active && active.file) this.lastPath = active.file.path;
    else if (!this.lastPath) {
      const f = this.app.workspace.getActiveFile();
      if (f) this.lastPath = f.path;
    }
    this.syncEditors();
    this.refreshViews();
  }

  syncEditors() {
    for (const view of this.markdownViews()) this.loadEditor(view);
  }

  itemsFor(path) {
    return (this.comments[path] || []).map((c) => ({
      id: c.id,
      from: c.from,
      to: c.to,
      orphaned: !!c.orphaned,
      visible: !c.resolved || this.settings.showResolved,
    }));
  }

  loadEditor(view, tries) {
    tries = tries || 0;
    const file = view.file;
    const cm = view.editor && view.editor.cm;
    if (!file || !cm || !cm.state.field(commentField, false)) return;

    if (view.__obcPath !== file.path) {
      // Wait until the editor actually holds the new file's text before anchoring
      if (typeof view.data === 'string' && cm.state.doc.toString() !== view.data && tries < 10) {
        window.setTimeout(() => this.loadEditor(view, tries + 1), 80);
        return;
      }
      this.reanchor(file.path, cm.state.doc.toString());
      view.__obcPath = file.path;
    }
    cm.dispatch({ effects: [setItems.of(this.itemsFor(file.path)), setActiveEffect.of(this.activeId)] });
  }

  reanchor(path, text) {
    const list = this.comments[path];
    if (!list) return;
    let dirty = false;
    for (const c of list) {
      if (!c.orphaned && c.to > c.from && text.slice(c.from, c.to) === c.quote) continue;
      const pos = findAnchor(text, c);
      if (pos == null) {
        if (!c.orphaned) dirty = true;
        c.orphaned = true;
      } else {
        c.from = pos;
        c.to = pos + c.quote.length;
        c.orphaned = false;
        dirty = true;
      }
    }
    if (dirty) this.requestSave();
  }

  /** Called on every document change: copy the mapped ranges from the editor into our data. */
  onDocChanged(cmView) {
    const view = this.markdownViews().find((v) => v.editor && v.editor.cm === cmView);
    if (!view || !view.file || view.__obcPath !== view.file.path) return;
    const list = this.comments[view.file.path];
    if (!list || !list.length) return;
    const val = cmView.state.field(commentField, false);
    if (!val) return;

    const byId = new Map(val.items.map((i) => [i.id, i]));
    const doc = cmView.state.doc;
    let dirty = false;
    let needRefresh = false;

    for (const c of list) {
      const it = byId.get(c.id);
      if (!it) continue;
      if (it.orphaned) {
        if (!c.orphaned) {
          c.orphaned = true;
          dirty = needRefresh = true;
        }
        continue;
      }
      const quote = doc.sliceString(it.from, it.to);
      if (c.from !== it.from || c.to !== it.to || c.quote !== quote) {
        if (c.quote !== quote) needRefresh = true;
        c.from = it.from;
        c.to = it.to;
        c.quote = quote;
        dirty = true;
      }
      c.prefix = doc.sliceString(Math.max(0, it.from - CONTEXT), it.from);
      c.suffix = doc.sliceString(it.to, Math.min(doc.length, it.to + CONTEXT));
    }
    if (dirty) this.requestSave();
    if (needRefresh) this.refreshViewsSoon();
  }

  /* ---------- active comment ---------- */

  setActive(id) {
    this.activeId = id;
    for (const view of this.markdownViews()) {
      const cm = view.editor && view.editor.cm;
      if (cm && cm.state.field(commentField, false)) cm.dispatch({ effects: setActiveEffect.of(id) });
    }
    for (const v of this.getSidebars()) v.setActiveCard(id);
  }

  onHighlightClick(id) {
    this.setActive(id);
    if (!this.app.workspace.getLeavesOfType(VIEW_TYPE).length) this.activateView();
  }

  goTo(c) {
    const view = this.markdownViews().find((v) => v.file && v.file.path === this.lastPath);
    const cm = view && view.editor && view.editor.cm;
    if (!cm || c.orphaned) return;
    cm.dispatch({ effects: [EditorView.scrollIntoView(c.from, { y: 'center' }), setActiveEffect.of(c.id)] });
  }

  /* ---------- selection button + composer ---------- */

  setupSelectionUI() {
    this.fab = document.body.createDiv({ cls: 'obc-fab' });
    setIcon(this.fab, 'message-square-plus');
    this.fab.setAttr('aria-label', 'Add comment');
    this.fab.addEventListener('mousedown', (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
    this.fab.addEventListener('click', () => {
      const view = this.getEditorView();
      if (!view) return;
      const sel = view.editor.cm.state.selection.main;
      if (!sel.empty) this.openComposer(view, sel.from, sel.to);
    });
    this.register(() => this.fab.remove());

    this.registerDomEvent(document, 'mouseup', (e) => {
      const t = e.target;
      if (t instanceof Node && (this.fab.contains(t) || (this.composer && this.composer.contains(t)))) return;
      window.setTimeout(() => this.updateFab(), 20);
    });
    this.registerDomEvent(document, 'keyup', (e) => {
      if (e.key === 'Escape') {
        this.hideFab();
        return;
      }
      if (this.composer) return;
      if (e.shiftKey || e.key.startsWith('Arrow') || e.key === 'a') window.setTimeout(() => this.updateFab(), 20);
    });
    this.registerDomEvent(document, 'scroll', () => this.hideFab(), true);
    this.registerDomEvent(document, 'mousedown', (e) => {
      if (!this.composer) return;
      const t = e.target;
      if (t instanceof Node && this.composer.contains(t)) return;
      const ta = this.composer.querySelector('textarea');
      if (ta && !ta.value.trim()) this.closeComposer();
    });
  }

  hideFab() {
    if (this.fab) this.fab.removeClass('is-visible');
  }

  updateFab() {
    this.hideFab();
    if (!this.commentMode || this.composer) return;
    const view = this.getEditorView();
    if (!view) return;
    const cm = view.editor.cm;
    if (!cm.dom.contains(document.activeElement)) return;
    const sel = cm.state.selection.main;
    if (sel.empty || !cm.state.sliceDoc(sel.from, sel.to).trim()) return;
    const rect = cm.coordsAtPos(sel.to);
    if (!rect) return;
    const left = Math.min(rect.right + 8, window.innerWidth - 44);
    const top = Math.min(rect.bottom + 6, window.innerHeight - 44);
    this.fab.style.left = left + 'px';
    this.fab.style.top = top + 'px';
    this.fab.addClass('is-visible');
  }

  openComposer(view, from, to) {
    this.hideFab();
    this.closeComposer();
    const cm = view.editor.cm;
    const quote = cm.state.sliceDoc(from, to);
    const rect = cm.coordsAtPos(to) || { left: 200, bottom: 200, top: 180 };
    cm.dispatch({ effects: setPendingEffect.of({ from, to }) });

    const el = document.body.createDiv({ cls: 'obc-composer' });
    el.createDiv({ cls: 'obc-composer-quote', text: truncate(quote, 140) });
    const ta = el.createEl('textarea', { attr: { placeholder: 'Add a comment…' } });
    const btns = el.createDiv({ cls: 'obc-composer-btns' });
    const cancel = btns.createEl('button', { text: 'Cancel' });
    const ok = btns.createEl('button', { text: 'Comment', cls: 'mod-cta' });

    const left = Math.max(8, Math.min(rect.left, window.innerWidth - 320));
    let top = rect.bottom + 8;
    el.style.left = left + 'px';
    el.style.top = top + 'px';
    const h = el.offsetHeight;
    if (top + h > window.innerHeight - 8) {
      top = Math.max(8, (rect.top || rect.bottom) - h - 8);
      el.style.top = top + 'px';
    }

    const submit = () => {
      const text = ta.value.trim();
      if (!text) return;
      const p = cm.state.field(commentField).pending || { from, to };
      this.closeComposer();
      this.addComment(view, p.from, p.to, text);
    };
    ok.addEventListener('click', submit);
    cancel.addEventListener('click', () => this.closeComposer());
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        submit();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        this.closeComposer();
      }
    });
    this.composer = el;
    this.composerCm = cm;
    window.setTimeout(() => ta.focus(), 0);
  }

  closeComposer() {
    if (this.composer) {
      this.composer.remove();
      this.composer = null;
    }
    if (this.composerCm) {
      try {
        this.composerCm.dispatch({ effects: setPendingEffect.of(null) });
      } catch (e) {
        /* view may be gone */
      }
      this.composerCm = null;
    }
  }

  /* ---------- comment CRUD ---------- */

  addComment(view, from, to, text) {
    const cm = view.editor.cm;
    const doc = cm.state.doc;
    const path = view.file.path;
    const c = {
      id: uid(),
      from,
      to,
      quote: doc.sliceString(from, to),
      prefix: doc.sliceString(Math.max(0, from - CONTEXT), from),
      suffix: doc.sliceString(to, Math.min(doc.length, to + CONTEXT)),
      created: Date.now(),
      resolved: false,
      orphaned: false,
      thread: [{ id: uid(), author: this.settings.author, text, ts: Date.now() }],
    };
    if (!this.comments[path]) this.comments[path] = [];
    this.comments[path].push(c);
    this.lastPath = path;
    this.requestSave();
    this.activeId = c.id;
    cm.dispatch({ selection: { anchor: to } });
    this.pushAll();
    this.activateView();
  }

  findComment(id) {
    for (const path of Object.keys(this.comments)) {
      const c = this.comments[path].find((x) => x.id === id);
      if (c) return { c, path };
    }
    return null;
  }

  addReply(id, text) {
    const f = this.findComment(id);
    if (!f) return;
    f.c.thread.push({ id: uid(), author: this.settings.author, text, ts: Date.now() });
    this.requestSave();
    this.refreshViews();
  }

  editMessage(id, msgId, text) {
    const f = this.findComment(id);
    if (!f) return;
    const m = f.c.thread.find((x) => x.id === msgId);
    if (!m) return;
    m.text = text;
    m.edited = true;
    this.requestSave();
    this.refreshViews();
  }

  deleteMessage(id, msgId) {
    const f = this.findComment(id);
    if (!f) return;
    f.c.thread = f.c.thread.filter((x) => x.id !== msgId);
    if (!f.c.thread.length) return this.deleteThread(id);
    this.requestSave();
    this.refreshViews();
  }

  deleteThread(id) {
    const f = this.findComment(id);
    if (!f) return;
    this.comments[f.path] = this.comments[f.path].filter((x) => x.id !== id);
    this.drafts.delete(id);
    if (this.activeId === id) this.activeId = null;
    this.requestSave();
    this.pushAll();
  }

  setResolved(id, resolved) {
    const f = this.findComment(id);
    if (!f) return;
    f.c.resolved = resolved;
    this.requestSave();
    this.pushAll();
  }

  /* ---------- sidebar ---------- */

  pushAll() {
    this.syncEditors();
    this.refreshViews();
  }

  getSidebars() {
    return this.app.workspace
      .getLeavesOfType(VIEW_TYPE)
      .map((l) => l.view)
      .filter((v) => v instanceof CommentsView);
  }

  refreshViews() {
    for (const v of this.getSidebars()) v.render();
  }

  async activateView() {
    const ws = this.app.workspace;
    let leaf = ws.getLeavesOfType(VIEW_TYPE)[0];
    if (!leaf) {
      leaf = ws.getRightLeaf(false);
      if (!leaf) return;
      await leaf.setViewState({ type: VIEW_TYPE, active: true });
    }
    ws.revealLeaf(leaf);
  }
}

/* =====================================================================
 * Sidebar view
 * ===================================================================== */

class CommentsView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
  }
  getViewType() {
    return VIEW_TYPE;
  }
  getDisplayText() {
    return 'Comments';
  }
  getIcon() {
    return 'message-square';
  }
  async onOpen() {
    this.contentEl.addClass('obc-view');
    this.render();
  }

  render() {
    const p = this.plugin;
    const el = this.contentEl;
    const prevList = el.querySelector('.obc-list');
    const st = prevList ? prevList.scrollTop : 0;
    el.empty();

    const bar = el.createDiv({ cls: 'obc-bar' });
    const modeBtn = bar.createEl('button', {
      text: p.commentMode ? 'Comment mode: on' : 'Comment mode: off',
      cls: p.commentMode ? 'mod-cta' : '',
    });
    modeBtn.addEventListener('click', () => p.setCommentMode(!p.commentMode));
    const lbl = bar.createEl('label');
    const cb = lbl.createEl('input', { type: 'checkbox' });
    cb.checked = p.settings.showResolved;
    lbl.createSpan({ text: 'Show resolved' });
    cb.addEventListener('change', () => {
      p.settings.showResolved = cb.checked;
      p.requestSave();
      p.pushAll();
    });

    const list = el.createDiv({ cls: 'obc-list' });
    const path = p.lastPath;
    if (!path) {
      list.createDiv({ cls: 'obc-empty', text: 'Open a note to see its comments.' });
      return;
    }

    const items = (p.comments[path] || [])
      .filter((c) => !c.resolved || p.settings.showResolved)
      .slice()
      .sort((a, b) => (a.orphaned === b.orphaned ? a.from - b.from : a.orphaned ? 1 : -1));

    if (!items.length) {
      list.createDiv({
        cls: 'obc-empty',
        text: 'No comments yet. Turn on comment mode, select some text, and click the comment button that appears.',
      });
    }
    for (const c of items) this.renderCard(list, c);
    list.scrollTop = st;
  }

  setActiveCard(id) {
    const cards = this.contentEl.querySelectorAll('.obc-card');
    let target = null;
    cards.forEach((card) => {
      const on = card.dataset.id === id;
      card.toggleClass('is-active', on);
      if (on) target = card;
    });
    if (target) target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  renderCard(parent, c) {
    const p = this.plugin;
    const card = parent.createDiv({ cls: 'obc-card' });
    card.dataset.id = c.id;
    card.toggleClass('is-active', p.activeId === c.id);
    card.toggleClass('is-resolved', !!c.resolved);

    card.createDiv({ cls: 'obc-quote', text: truncate(c.quote, 160) });
    if (c.orphaned) card.createDiv({ cls: 'obc-orphan', text: 'The commented text was deleted or changed.' });

    c.thread.forEach((m, idx) => {
      const row = card.createDiv({ cls: 'obc-msg' });
      const head = row.createDiv({ cls: 'obc-msg-head' });
      head.createSpan({ cls: 'obc-author', text: m.author });
      head.createSpan({ cls: 'obc-time', text: relTime(m.ts) + (m.edited ? ' · edited' : '') });
      const more = head.createSpan({ cls: 'obc-icon-btn' });
      setIcon(more, 'more-horizontal');
      const body = row.createDiv({ cls: 'obc-msg-body', text: m.text });
      more.addEventListener('click', (e) => {
        e.stopPropagation();
        const menu = new Menu();
        menu.addItem((i) =>
          i
            .setTitle('Edit')
            .setIcon('pencil')
            .onClick(() => this.startEdit(c, m, body))
        );
        menu.addItem((i) =>
          i
            .setTitle(idx === 0 ? 'Delete thread' : 'Delete reply')
            .setIcon('trash')
            .onClick(() => (idx === 0 ? p.deleteThread(c.id) : p.deleteMessage(c.id, m.id)))
        );
        menu.showAtMouseEvent(e);
      });
    });

    const reply = card.createDiv({ cls: 'obc-reply' });
    const ta = reply.createEl('textarea', { attr: { placeholder: 'Reply…', rows: '1' } });
    ta.value = p.drafts.get(c.id) || '';
    reply.toggleClass('has-text', !!ta.value);
    const send = () => {
      const text = ta.value.trim();
      if (!text) return;
      p.drafts.delete(c.id);
      p.addReply(c.id, text);
    };
    ta.addEventListener('input', () => {
      p.drafts.set(c.id, ta.value);
      reply.toggleClass('has-text', !!ta.value);
    });
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        send();
      }
    });
    const btns = reply.createDiv({ cls: 'obc-reply-btns' });
    const resolveBtn = btns.createEl('button', { text: c.resolved ? 'Reopen' : 'Resolve' });
    resolveBtn.addEventListener('click', () => p.setResolved(c.id, !c.resolved));
    const replyBtn = btns.createEl('button', { text: 'Reply', cls: 'mod-cta' });
    replyBtn.addEventListener('click', send);

    card.addEventListener('click', (e) => {
      const t = e.target;
      if (t instanceof HTMLElement && t.closest('textarea,button,.obc-icon-btn')) return;
      p.setActive(c.id);
      p.goTo(c);
    });
  }

  startEdit(c, m, bodyEl) {
    bodyEl.empty();
    const ta = bodyEl.createEl('textarea');
    ta.value = m.text;
    const btns = bodyEl.createDiv({ cls: 'obc-edit-btns' });
    const cancel = btns.createEl('button', { text: 'Cancel' });
    const save = btns.createEl('button', { text: 'Save', cls: 'mod-cta' });
    cancel.addEventListener('click', () => this.render());
    save.addEventListener('click', () => {
      const text = ta.value.trim();
      if (text) this.plugin.editMessage(c.id, m.id, text);
    });
    ta.focus();
  }
}

/* =====================================================================
 * Settings
 * ===================================================================== */

class InlineCommentsSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    const p = this.plugin;
    containerEl.empty();

    new Setting(containerEl)
      .setName('Your name')
      .setDesc('Shown as the author of your comments and replies.')
      .addText((t) =>
        t.setValue(p.settings.author).onChange((v) => {
          p.settings.author = v.trim() || 'Me';
          p.requestSave();
        })
      );

    new Setting(containerEl)
      .setName('Show resolved comments')
      .setDesc('Keep resolved threads visible in the sidebar and highlighted in the note.')
      .addToggle((t) =>
        t.setValue(p.settings.showResolved).onChange((v) => {
          p.settings.showResolved = v;
          p.requestSave();
          p.pushAll();
        })
      );

    new Setting(containerEl)
      .setName('Highlight colour')
      .addColorPicker((c) =>
        c.setValue(p.settings.color).onChange((v) => {
          p.settings.color = v;
          p.applyColor();
          p.requestSave();
        })
      );

    containerEl.createEl('p', {
      cls: 'setting-item-description',
      text: 'Comments are stored in this plugin\'s data.json (not inside your notes), so your markdown stays clean. Highlights show in Live Preview and Source mode.',
    });
  }
}

module.exports = InlineCommentsPlugin;
