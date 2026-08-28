/**
 * Run the joint-box labelling page against a stub DOM.
 *
 * There is no browser in CI, but the failure this guards against does not need
 * one: the page wires itself to two dozen element ids at load time, and a typo
 * in any of them throws before the reviewer draws a single box. Executing the
 * real script against stubs catches that, and lets the export/import round-trip
 * be checked on actual coordinates.
 *
 *   node tests/js/joint_box_app_smoke.mjs <path-to-label_boxes.html>
 */
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const htmlPath = process.argv[2];
if (!htmlPath) { console.error('usage: joint_box_app_smoke.mjs <label_boxes.html>'); process.exit(2); }
const html = readFileSync(htmlPath, 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];

const ids = [...html.matchAll(/id="([\w-]+)"/g)].map(m => m[1]);
const store = new Map();
const globalListeners = {};

function makeEl(id = '') {
  const el = {
    id, tagName: 'DIV', textContent: '', innerHTML: '', value: '', files: [],
    style: {}, children: [], _listeners: {},
    classList: {
      _s: new Set(),
      add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
      contains(c) { return this._s.has(c); },
      toggle(c, on) { const has = this._s.has(c); const want = on === undefined ? !has : on;
        want ? this._s.add(c) : this._s.delete(c); return want; },
    },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); },
    removeEventListener() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 800, height: 600 }; },
    setPointerCapture() {}, click() { if (this.onclick) this.onclick({ target: this }); },
    getContext() { return ctx2d; },
    dispatch(type, ev) { (this._listeners[type] || []).forEach(fn => fn(ev)); },
  };
  return el;
}

const ctx2d = new Proxy({}, {
  get: (_t, prop) => {
    if (prop === 'canvas') return undefined;
    return () => {};
  },
  set: () => true,
});

const elements = new Map(ids.map(id => [id, makeEl(id)]));
const canvas = elements.get('cv');
canvas.tagName = 'CANVAS'; canvas.width = 0; canvas.height = 0;

const document = {
  querySelector(sel) {
    if (sel.startsWith('#')) {
      const id = sel.slice(1);
      if (!elements.has(id)) throw new Error(`page asked for #${id}, which is not in the markup`);
      return elements.get(id);
    }
    return makeEl();
  },
  querySelectorAll() { const a = []; a.forEach = Array.prototype.forEach.bind(a); return a; },
  createElement(tag) { const el = makeEl(); el.tagName = tag.toUpperCase(); return el; },
};

class StubImage {
  constructor() { this.complete = false; this.naturalWidth = 0; this.naturalHeight = 0; }
  set src(v) {
    this._src = v;
    setTimeout(() => {
      this.complete = true; this.naturalWidth = 120; this.naturalHeight = 90;
      if (this.onload) this.onload();
    }, 0);
  }
  get src() { return this._src; }
}

let downloaded = null;
const sandbox = {
  document,
  Image: StubImage,
  console,
  setTimeout, clearTimeout, Date, JSON, Math, Object, Array, Set, String, Number, Blob: class {
    constructor(parts) { this.parts = parts; }
  },
  URL: { createObjectURL(b) { downloaded = b.parts.join(''); return 'blob:stub'; },
         revokeObjectURL() {} },
  FileReader: class {
    readAsText(file) {
      this.result = file.contents;
      if (this.onload) this.onload();
    }
  },
  confirm: () => true,
  localStorage: {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, v),
    removeItem: k => store.delete(k),
  },
  addEventListener(type, fn) { (globalListeners[type] ||= []).push(fn); },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
vm.runInContext(script, sandbox, { filename: 'label_boxes.js' });

// let the stubbed image load resolve
await new Promise(r => setTimeout(r, 5));

const fail = m => { console.error('FAIL: ' + m); process.exit(1); };
const press = key => {
  const event = { key, target: { tagName: 'BODY' }, preventDefault() {} };
  for (const listener of globalListeners.keydown || []) listener(event);
};

// 1. the page reached the end of its wiring and knows how many crops it has
const total = vm.runInContext('rows.length', sandbox);
if (!(total > 0)) fail('no rows reached the page');

if (total < 4) fail(`fixture needs at least four crops, got ${total}`);

// 2. An AI draft is deliberately unreviewed even though it already has boxes.
// Loading it must keep the blank status, and the todo filter must include it.
const draftPaths = vm.runInContext('[rows[0].crop_path, rows[1].crop_path]', sandbox);
const datasetId = vm.runInContext('DATA.dataset_id', sandbox);
// Tên lớp đọc TỪ TRANG, không viết cứng: cùng một tool phục vụ cả bộ mối hàn
// (solder_joint) lẫn bộ thân linh kiện (component), và một test chỉ chạy được
// với một trong hai thì nó kiểm cấu hình chứ không kiểm code.
const firstClass = vm.runInContext('CLASSES[0].name', sandbox);
const draft = {
  schema: 'aoi-joint-boxes/1.0',
  dataset_id: datasetId,
  crops: {
    [draftPaths[0]]: {
      status: '', notes: 'AI proposal',
      boxes: [{ cls: firstClass, x: 10.4, y: 20.6, w: 30.2, h: 15.8 }],
    },
    [draftPaths[1]]: {
      status: '', notes: '',
      boxes: [{ cls: firstClass, x: 3, y: 4, w: 12, h: 9 }],
    },
  },
};
const fileInput = elements.get('file');
fileInput.files = [{ contents: JSON.stringify(draft) }];
fileInput.onchange({ target: fileInput });

const loadedDraft = vm.runInContext(
  `JSON.parse(JSON.stringify(state[rows[0].crop_path]))`, sandbox,
);
if (loadedDraft.status !== '') fail('AI draft was marked reviewed while loading');
if (loadedDraft.boxes.length !== 1 || loadedDraft.boxes[0].cls !== 0) {
  fail('AI draft box/class was not loaded intact: ' + JSON.stringify(loadedDraft));
}
vm.runInContext(`$('#filter').value = 'todo'; idx = 0; show();`, sandbox);
const todoPaths = vm.runInContext('visible().map(r => r.crop_path)', sandbox);
if (!todoPaths.includes(draftPaths[0]) || !todoPaths.includes(draftPaths[1])) {
  fail('blank-status AI drafts are missing from the todo filter');
}

// Enter approves an AI proposal without changing its boxes. C approves the next
// crop as clean and therefore removes the proposed box.
press('Enter');
const enterResult = vm.runInContext(
  `JSON.parse(JSON.stringify(state[rows[0].crop_path]))`, sandbox,
);
if (enterResult.status !== 'verified' || enterResult.boxes.length !== 1) {
  fail('Enter did not approve and preserve the AI box');
}
press('c');
const cleanResult = vm.runInContext(
  `JSON.parse(JSON.stringify(state[rows[1].crop_path]))`, sandbox,
);
if (cleanResult.status !== 'verified' || cleanResult.boxes.length !== 0) {
  fail('C did not approve the crop as clean');
}
const remainingTodo = vm.runInContext('visible().map(r => r.crop_path)', sandbox);
if (remainingTodo.includes(draftPaths[0]) || remainingTodo.includes(draftPaths[1])) {
  fail('reviewed drafts remained in the todo filter');
}

// 3. Export carries only reviewed records and writes crop-pixel coordinates.
vm.runInContext(`
  st(rows[2].crop_path).status = 'skipped';
  $('#rev').value = 'qnn';
`, sandbox);
vm.runInContext(`$('#save').onclick()`, sandbox);
if (!downloaded) fail('export produced nothing');
const payload = JSON.parse(downloaded);

if (payload.schema !== 'aoi-joint-boxes/1.0') fail('wrong schema tag: ' + payload.schema);
if (payload.reviewer_id !== 'qnn') fail('reviewer not carried into the export');
if (payload.coordinate_space !== 'crop_pixels_top_left_origin') fail('coordinate space undeclared');

const first = vm.runInContext('rows[0].crop_path', sandbox);
const rec = payload.crops[first];
if (!rec || rec.status !== 'verified') fail('verified crop missing from export');
if (rec.boxes.length !== 1) fail('box not exported');
if (rec.boxes[0].cls !== firstClass) fail('class index exported instead of name: ' + rec.boxes[0].cls);
const b = rec.boxes[0];
if (b.x !== 10 || b.y !== 21 || b.w !== 30 || b.h !== 16) fail('coordinates not rounded as expected: ' + JSON.stringify(b));

const second = vm.runInContext('rows[1].crop_path', sandbox);
if (payload.crops[second].boxes.length !== 0) fail('clean crop should export zero boxes');
if (payload.crops[second].status !== 'verified') fail('clean crop must be verified, not blank');

// 4. an untouched crop must never appear: silence is not a label
const untouched = vm.runInContext('rows[rows.length-1].crop_path', sandbox);
if (untouched in payload.crops) fail('unreviewed crop leaked into the export');

console.log(`ok: ${total} rows, export carries ${Object.keys(payload.crops).length} reviewed crops`);
