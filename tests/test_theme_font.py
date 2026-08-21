"""Font của app phải là Montserrat, và phải tự phục vụ được.

Hai chuyện khác nhau, và test này giữ cả hai:

1. **Montserrat, không phải font hệ thống.** Đây là quy ước chung của mọi UI
   trong dự án. Ghi "Segoe UI" thẳng vào chỗ nào đó là mỗi máy hiện một kiểu.

2. **Không gọi ra CDN.** App chạy ở xưởng. Một request tới
   `fonts.googleapis.com` ở đó không phải là "chậm một chút" — nó là trang
   hiện bằng font dự phòng, hoặc treo chờ timeout. Font phải nằm trong repo.

Chỗ dễ hỏng nhất không phải khai báo mà là **file font biến mất**: gitignore
bắt nhầm `*.woff2`, hoặc ai đó dọn `app/static/`. Khi đó config vẫn đúng, test
cũ vẫn xanh, và app im lặng rơi về Arial. Nên ở đây kiểm cả file thật.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / ".streamlit" / "config.toml"
STYLES_PATH = PROJECT_ROOT / "app" / "assets" / "styles.css"

#: Streamlit chỉ phục vụ thư mục `static` NẰM CẠNH file script chính.
STATIC_ROOT = PROJECT_ROOT / "app" / "static"


def _config() -> dict:
    return tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_theme_asks_for_montserrat_first() -> None:
    font = _config()["theme"]["font"]
    assert font.split(",")[0].strip() == "Montserrat", (
        f"font đang là {font!r}; quy ước dự án là Montserrat đứng đầu"
    )
    assert "sans-serif" in font, "phải có dự phòng chung, phòng khi file font mất"


def test_static_serving_is_on_or_the_font_urls_are_dead_links() -> None:
    """Không có cờ này thì Streamlit trả 404 cho mọi URL `app/static/...`,
    và font im lặng rơi về dự phòng."""

    assert _config()["server"]["enableStaticServing"] is True


def test_every_declared_font_file_actually_exists() -> None:
    faces = _config()["theme"]["fontFaces"]
    assert faces, "không có [[theme.fontFaces]] nào — font sẽ không tải được"

    missing = []
    for face in faces:
        url = face["url"]
        assert url.startswith("app/static/"), (
            f"{url!r} không nằm trong thư mục static; Streamlit sẽ không phục vụ nó"
        )
        # "app/static/x" -> <PROJECT>/app/static/x
        path = STATIC_ROOT / Path(url).relative_to("app/static")
        if not path.is_file():
            missing.append(url)
    assert not missing, f"khai báo nhưng thiếu file: {missing}"


def test_no_font_is_fetched_from_the_network() -> None:
    """Một URL http(s) trong fontFaces là đúng thứ mục 2 của docstring cấm."""

    for face in _config()["theme"]["fontFaces"]:
        assert not face["url"].startswith(("http://", "https://")), (
            f"{face['url']} gọi ra mạng; app phải chạy được khi mất mạng"
        )
    assert "fonts.googleapis.com" not in STYLES_PATH.read_text(encoding="utf-8")


def test_vietnamese_glyphs_are_covered() -> None:
    """Không có subset `vietnamese` thì dấu tiếng Việt rơi sang font dự phòng
    — chữ vẫn đọc được nhưng lệch nét, và đó là lỗi hay bị bỏ qua nhất."""

    faces = _config()["theme"]["fontFaces"]
    vietnamese = [face for face in faces if "vietnamese" in face["url"]]
    assert vietnamese, "thiếu subset tiếng Việt"

    for face in vietnamese:
        # U+1EA0-1EF9 là khối chứa phần lớn nguyên âm có dấu của tiếng Việt.
        assert "1EA0" in face.get("unicodeRange", "").upper(), (
            f"{face['url']} không khai unicode-range tiếng Việt, trình duyệt "
            "sẽ không biết khi nào cần tới nó"
        )


def test_the_stylesheet_agrees_with_the_theme() -> None:
    """Các khối HTML tự viết nằm ngoài cây component của Streamlit, nên chúng
    không thừa hưởng `theme.font` và phải được chỉ định riêng."""

    css = STYLES_PATH.read_text(encoding="utf-8")
    assert "font-family: Montserrat" in css
    # Font hệ thống chỉ được xuất hiện như DỰ PHÒNG, không bao giờ đứng đầu.
    for line in css.splitlines():
        if "font-family:" in line and "Montserrat" not in line:
            stack = line.split("font-family:")[1].strip().rstrip(";")
            assert stack.split(",")[0].strip().strip('"') in {
                "Consolas", "monospace",
            }, f"font-family không phải Montserrat và không phải monospace: {line.strip()}"
