/* Regenerates the sources table in README.md directly from js/mechanisms.js, so
   the README and the app can never disagree about a date.
   Run:  node scripts/gen_sources_table.js     (from assignment-8/) */
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..');

const src = fs.readFileSync(path.join(ROOT, 'js', 'mechanisms.js'), 'utf8');
const { MECHANISMS, prettyDate } = new Function(
  src + '\nreturn { MECHANISMS, prettyDate };')();

const cell = s => String(s).replace(/\|/g, '\\|').replace(/\n/g, ' ');

const rows = MECHANISMS.map((m, i) => {
  const link = `[${cell(m.sourceLabel)}](${m.sourceUrl})`;
  const kind = m.arxiv ? 'arXiv v1' : 'release/post';
  return `| ${i + 1} | \`${m.date}\` | ${cell(m.name)} | ${kind} | ${link} |`;
}).join('\n');

const table = [
  '| # | Date | Mechanism | Date type | Primary source |',
  '|---|------|-----------|-----------|----------------|',
  rows
].join('\n');

const basis = MECHANISMS.filter(m => !m.arxiv).map(m =>
  `**${prettyDate(m.date)} — ${cell(m.name)}**\n\n${cell(m.dateBasis)}\n\n<${m.sourceUrl}>`
).join('\n\n---\n\n');

const README = path.join(ROOT, 'README.md');
let text = fs.readFileSync(README, 'utf8');

function replaceBlock(name, content) {
  const open = `<!-- BEGIN ${name} -->`, close = `<!-- END ${name} -->`;
  const re = new RegExp(`${open}[\\s\\S]*?${close}`);
  if (!re.test(text)) throw new Error(`marker ${name} not found in README.md`);
  text = text.replace(re, `${open}\n${content}\n${close}`);
}

replaceBlock('SOURCES', table);
replaceBlock('NOPAPER', basis);
fs.writeFileSync(README, text);
console.log(`README.md updated: ${MECHANISMS.length} rows, ` +
            `${MECHANISMS.filter(m => m.arxiv).length} arXiv-backed.`);
