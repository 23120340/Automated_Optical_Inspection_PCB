/** Exercise the package-only migration and unknown-label safety contract. */
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const htmlPath = process.argv[2];
if (!htmlPath) {
  console.error('usage: package_label_app_smoke.mjs <label_packages.html>');
  process.exit(2);
}
const html = readFileSync(htmlPath, 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const dataMatch = html.match(/const DATA = (.*?);\r?\nconst CLASSES/s);
if (!dataMatch) throw new Error('cannot find embedded DATA payload');
const data = JSON.parse(dataMatch[1]);
const alias = data.migration_aliases?.[0];
if (!alias) throw new Error('package page has no migration alias');
if (data.classes.length !== 7 || data.unknown_class !== 'unknown') {
  throw new Error('package taxonomy/editor sentinel contract is missing');
}

const ids = [...html.matchAll(/id="([\w-]+)"/g)].map(match => match[1]);
const store = new Map();
const globalListeners = {};
const migratedPath = data.rows[0].crop_path;
const geometry = { x: 10.25, y: 20.5, w: 30.75, h: 15.125 };
const oldKey = 'jointbox:' + alias.dataset_id;
store.set(oldKey, JSON.stringify({
  [migratedPath]: {
    status: 'verified', notes: 'body reviewed',
    boxes: [{ cls: 0, ...geometry }],
  },
}));
store.set(oldKey + ':rev', 'reviewer-before-package');

const ctx2d = new Proxy({}, {
  get: (_target, property) => property === 'canvas' ? undefined : (() => {}),
  set: () => true,
});
function makeEl(id = '') {
  return {
    id, tagName: 'DIV', textContent: '', innerHTML: '', value: '', files: [],
    style: {}, children: [], _listeners: {},
    classList: {
      _values: new Set(),
      add(value) { this._values.add(value); },
      remove(value) { this._values.delete(value); },
      contains(value) { return this._values.has(value); },
      toggle(value, force) {
        const wanted = force === undefined ? !this._values.has(value) : force;
        wanted ? this._values.add(value) : this._values.delete(value);
        return wanted;
      },
    },
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(type, listener) { (this._listeners[type] ||= []).push(listener); },
    removeEventListener() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 600 }; },
    setPointerCapture() {},
    click() { if (this.onclick) this.onclick({ target: this }); },
    getContext() { return ctx2d; },
  };
}
const elements = new Map(ids.map(id => [id, makeEl(id)]));
const canvas = elements.get('cv');
canvas.tagName = 'CANVAS';
canvas.width = 0; canvas.height = 0;
const document = {
  querySelector(selector) {
    if (selector.startsWith('#')) {
      const id = selector.slice(1);
      if (!elements.has(id)) throw new Error(`missing DOM element #${id}`);
      return elements.get(id);
    }
    return makeEl();
  },
  querySelectorAll() {
    const values = [];
    values.forEach = Array.prototype.forEach.bind(values);
    return values;
  },
  createElement(tag) {
    const element = makeEl();
    element.tagName = tag.toUpperCase();
    return element;
  },
};
class StubImage {
  constructor() { this.complete = false; this.naturalWidth = 0; this.naturalHeight = 0; }
  set src(value) {
    this._src = value;
    setTimeout(() => {
      this.complete = true; this.naturalWidth = 120; this.naturalHeight = 90;
      if (this.onload) this.onload();
    }, 0);
  }
  get src() { return this._src; }
}

let downloaded = null;
const sandbox = {
  document, Image: StubImage, console, setTimeout, clearTimeout,
  Date, JSON, Math, Object, Array, Set, String, Number,
  Blob: class { constructor(parts) { this.parts = parts; } },
  URL: {
    createObjectURL(blob) { downloaded = blob.parts.join(''); return 'blob:stub'; },
    revokeObjectURL() {},
  },
  FileReader: class {}, confirm: () => true,
  localStorage: {
    getItem: key => store.has(key) ? store.get(key) : null,
    setItem: (key, value) => store.set(key, value),
    removeItem: key => store.delete(key),
  },
  addEventListener(type, listener) { (globalListeners[type] ||= []).push(listener); },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(script, sandbox, { filename: 'label_packages.js' });
await new Promise(resolve => setTimeout(resolve, 5));

const fail = message => { console.error('FAIL: ' + message); process.exit(1); };
const evaluate = source => vm.runInContext(source, sandbox);
const snapshot = () => evaluate(
  `JSON.parse(JSON.stringify(state[${JSON.stringify(migratedPath)}]))`
);
const press = key => {
  const event = { key, target: { tagName: 'BODY' }, preventDefault() {} };
  for (const listener of globalListeners.keydown || []) listener(event);
};

const migrated = snapshot();
if (migrated.status !== '' || migrated.source_status !== 'verified') {
  fail('old review status was reinterpreted as a reviewed package');
}
if (migrated.boxes.length !== 1 || migrated.boxes[0].cls !== -1) {
  fail('old class index was not reset to unknown');
}
for (const [field, value] of Object.entries(geometry)) {
  if (migrated.boxes[0][field] !== value) fail(`geometry ${field} changed during migration`);
}
if (!store.has(oldKey) || !store.has('jointbox:' + data.dataset_id)) {
  fail('migration did not preserve the old key and create the new key');
}
const receipt = JSON.parse(store.get('jointbox:' + data.dataset_id + ':migration'));
if (receipt.source_crops_semantic_sha256 !== alias.source_crops_semantic_sha256 ||
    receipt.box_geometry_semantic_sha256 !== alias.box_geometry_semantic_sha256) {
  fail('migration receipt lost semantic hashes');
}
if (elements.get('rev').value !== 'reviewer-before-package') {
  fail('reviewer id was not migrated');
}

// Enter must not approve an unresolved package, and an injected invalid state
// must not produce even a partial download.
press('Enter');
if (snapshot().status !== '') fail('Enter approved an unknown package');
evaluate(`state[${JSON.stringify(migratedPath)}].status='verified'`);
elements.get('save').onclick();
if (downloaded !== null) fail('export emitted a payload containing unknown');

// Resolve the exact selected box through the same 1–7 shortcut used by a
// reviewer, then approval and export are allowed.
evaluate(`state[${JSON.stringify(migratedPath)}].status=''; selBox=0`);
press('1');
if (snapshot().boxes[0].cls !== 0 || snapshot().boxes[0].needs_review !== false) {
  fail('numeric shortcut did not resolve the selected unknown box');
}
press('Enter');
if (snapshot().status !== 'verified') fail('resolved package could not be approved');
elements.get('save').onclick();
if (!downloaded) fail('resolved package export produced nothing');
const payload = JSON.parse(downloaded);
const exported = payload.crops[migratedPath];
if (!exported || exported.boxes[0].cls !== data.classes[0].name) {
  fail('export did not use the seven-class slug');
}
if (JSON.stringify(payload).includes('"cls":"unknown"') ||
    JSON.stringify(payload).includes('"cls":-1')) {
  fail('unknown sentinel leaked into export');
}

console.log('ok: package migration preserves geometry and blocks unknown export');
