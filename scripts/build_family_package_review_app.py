"""Trang duyệt nhãn HỌ + GÓI cho tập kiểm 750 box.

Nhãn đã được gán sẵn **bằng mắt**, không bằng 6.1 — tập này dùng để đo 6.1 nên
điền sẵn bằng chính nó thì phép đo tự xác nhận. Trang này để người duyệt sửa
lại, và mọi ô ``XEM_KY`` là chỗ máy đã tự nhận là không đọc được.

Ảnh không nhúng vào trang (44 MB): trang đọc thư mục ``crops/`` bên cạnh nó,
nên phải mở trang **tại chỗ** trong thư mục tập kiểm.

    python scripts/build_family_package_review_app.py --set <thư mục tập kiểm>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAMILIES = [
    "resistor", "capacitor", "ic", "connector", "diode", "led",
    "discrete_semiconductor", "magnetic", "timing", "protection", "relay",
    "switch_control", "display", "acoustic", "battery_power_input",
    "false_crop_background", "XEM_KY",
]
PACKAGES = [
    "hai_chan", "tru_dung", "goi_nho", "ic_hai_ben", "ic_bon_ben",
    "ic_khong_chan", "connector", "ngoai_taxonomy", "XEM_KY",
]

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Duyệt nhãn họ + gói — __N__ box</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;background:#14161a;color:#e8eaed;
      font:14px/1.5 Montserrat,system-ui,sans-serif}
 header{position:sticky;top:0;z-index:9;background:#1b1e24;padding:10px 14px;
        border-bottom:1px solid #2c313a;display:flex;gap:14px;align-items:center;
        flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600}
 .pill{background:#232830;border:1px solid #333a45;border-radius:99px;
       padding:3px 10px;font-size:12px}
 .warn{background:#3a2a12;border-color:#6b4d1a;color:#ffcf7a}
 button{background:#232830;color:#e8eaed;border:1px solid #39404c;
        border-radius:6px;padding:6px 12px;cursor:pointer;font:inherit}
 button:hover{background:#2c323c}
 main{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
      gap:12px;padding:14px}
 .card{background:#1b1e24;border:1px solid #2c313a;border-radius:9px;padding:9px}
 .card.xk{border-color:#6b4d1a}
 .card.done{border-color:#2f6b3a}
 .imgwrap{background:#0d0f12;border-radius:6px;height:150px;display:flex;
          align-items:center;justify-content:center;overflow:hidden}
 .imgwrap img{max-width:100%;max-height:100%;image-rendering:pixelated}
 .meta{font-size:11px;color:#98a0ad;margin:6px 0 3px;display:flex;
       justify-content:space-between;gap:6px}
 .note{font-size:11px;color:#c2c8d2;min-height:30px;margin-bottom:6px}
 select{width:100%;background:#232830;color:#e8eaed;border:1px solid #39404c;
        border-radius:5px;padding:4px;font:inherit;font-size:12px;margin-bottom:4px}
 select.xk{border-color:#6b4d1a;color:#ffcf7a}
 label{font-size:10px;color:#7d8694;text-transform:uppercase;letter-spacing:.4px}
</style>
<header>
  <h1>Duyệt nhãn họ + gói</h1>
  <span class="pill" id="count"></span>
  <span class="pill warn" id="xk"></span>
  <label style="text-transform:none;color:#98a0ad">
    <input type="checkbox" id="only"> chỉ hiện ô cần xem kỹ
  </label>
  <button id="save">Tải JSON đã sửa</button>
  <span class="pill">Nhãn điền sẵn do NHÌN ẢNH, không do model 6.1</span>
</header>
<main id="grid"></main>
<script>
const ITEMS = __DATA__;
const FAMILIES = __FAMILIES__;
const PACKAGES = __PACKAGES__;
const grid = document.getElementById('grid');

function options(list, chosen){
  return list.map(v => `<option${v===chosen?' selected':''}>${v}</option>`).join('');
}

function render(){
  const only = document.getElementById('only').checked;
  grid.innerHTML = '';
  let shown = 0;
  for (const it of ITEMS){
    const unsure = it.family === 'XEM_KY' || it.package === 'XEM_KY';
    if (only && !unsure) continue;
    shown++;
    const card = document.createElement('div');
    card.className = 'card ' + (unsure ? 'xk' : 'done');
    card.innerHTML = `
      <div class="imgwrap"><img loading="lazy" src="crops/${String(it.id).padStart(4,'0')}.png"></div>
      <div class="meta"><span>#${it.id}</span><span>${it.long_side}px · ${it.stratum}</span></div>
      <div class="note">${it.note}</div>
      <label>họ</label>
      <select class="${it.family==='XEM_KY'?'xk':''}" data-id="${it.id}" data-k="family">
        ${options(FAMILIES, it.family)}</select>
      <label>gói</label>
      <select class="${it.package==='XEM_KY'?'xk':''}" data-id="${it.id}" data-k="package">
        ${options(PACKAGES, it.package)}</select>`;
    grid.appendChild(card);
  }
  document.getElementById('count').textContent =
    `${shown} / ${ITEMS.length} ô đang hiện`;
  const left = ITEMS.filter(i => i.family==='XEM_KY' || i.package==='XEM_KY').length;
  document.getElementById('xk').textContent = `${left} ô còn XEM_KY`;
}

grid.addEventListener('change', e => {
  const sel = e.target.closest('select');
  if (!sel) return;
  const item = ITEMS.find(i => i.id === +sel.dataset.id);
  item[sel.dataset.k] = sel.value;
  // Vẽ lại cả lưới thì mất chỗ đang cuộn; chỉ đổi màu đúng ô vừa sửa.
  sel.classList.toggle('xk', sel.value === 'XEM_KY');
  const card = sel.closest('.card');
  const unsure = item.family === 'XEM_KY' || item.package === 'XEM_KY';
  card.className = 'card ' + (unsure ? 'xk' : 'done');
  const left = ITEMS.filter(i => i.family==='XEM_KY' || i.package==='XEM_KY').length;
  document.getElementById('xk').textContent = `${left} ô còn XEM_KY`;
});

document.getElementById('only').addEventListener('change', render);
document.getElementById('save').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({
    schema: 'aoi-family-package-review/1.0',
    reviewed_at: new Date().toISOString(),
    items: ITEMS,
  }, null, 1)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'family_package_reviewed.json';
  a.click();
});

render();
</script>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="folder", type=Path, required=True)
    args = parser.parse_args(argv)

    root = (PROJECT_ROOT / args.folder).resolve()
    payload = json.loads((root / "prelabels.json").read_text(encoding="utf-8"))
    items = [
        {
            "id": item["id"],
            "tile": item["tile"],
            "box_index": item["box_index"],
            "long_side": round(item["long_side"]),
            "stratum": item["stratum_reason"],
            "family": item["family_prelabel"],
            "package": item["package_prelabel"],
            "note": item["note"],
        }
        for item in sorted(payload["items"], key=lambda i: i["id"])
    ]
    html = (
        TEMPLATE
        .replace("__DATA__", json.dumps(items, ensure_ascii=False))
        .replace("__FAMILIES__", json.dumps(FAMILIES))
        .replace("__PACKAGES__", json.dumps(PACKAGES))
        .replace("__N__", str(len(items)))
    )
    target = root / "review_family_package.html"
    target.write_text(html, encoding="utf-8")
    unsure = sum(
        1 for i in items if i["family"] == "XEM_KY" or i["package"] == "XEM_KY"
    )
    print(f"{len(items)} box, {unsure} ô còn XEM_KY")
    print(f"ghi -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
