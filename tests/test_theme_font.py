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


# --------------------------------------------------------------------------
# Sidebar nền tối: chữ phải đọc được
#
# Lỗi thật, 2026-08-22: selectbox và đầu expander trong sidebar là chữ gần
# trắng trên nền TRẮNG. Nguyên nhân không phải chọn màu xấu mà là hai nửa của
# cùng một quyết định nằm ở hai nơi: nền widget do `secondaryBackgroundColor`
# quyết định (sáng, cho toàn app), còn màu chữ bị một luật CSS quét
# `[data-testid="stSidebar"] * { color: ... }` ép sang sáng.
#
# Nay cả hai khai ở `[theme.sidebar]`, và test dưới đo tỉ lệ tương phản thật
# thay vì tin vào mắt người viết CSS.
# --------------------------------------------------------------------------


def _luminance(hex_colour: str) -> float:
    """Độ sáng tương đối theo WCAG."""

    value = hex_colour.lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
              for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    a, b = _luminance(foreground), _luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def test_the_sidebar_declares_its_own_palette() -> None:
    """Không có section này thì widget trong sidebar lấy màu sáng của toàn app,
    và mọi cách sửa còn lại chỉ là đè CSS lên từng widget một."""

    assert "sidebar" in _config()["theme"], "thiếu [theme.sidebar]"


def test_sidebar_text_is_readable_on_both_of_its_backgrounds() -> None:
    """Hai nền, không phải một. `backgroundColor` là nền sidebar,
    `secondaryBackgroundColor` là nền của widget bên trong nó — selectbox, đầu
    expander, ô nhập. Chính cái thứ hai là chỗ đã hỏng."""

    sidebar = _config()["theme"]["sidebar"]
    text = sidebar["textColor"]
    for key in ("backgroundColor", "secondaryBackgroundColor"):
        ratio = _contrast(text, sidebar[key])
        assert ratio >= 4.5, (
            f"chữ {text} trên {key} {sidebar[key]} chỉ tương phản {ratio:.2f}:1, "
            "dưới mức 4.5:1 của WCAG AA — đây đúng là kiểu lỗi 'không đọc được chữ'"
        )


def test_the_sidebar_background_is_actually_dark() -> None:
    """Nếu ai đó đổi nó sang màu sáng, test tương phản ở trên vẫn có thể xanh
    (chữ tối trên nền sáng cũng đọc được) nhưng sidebar sẽ không còn khớp phần
    còn lại của thiết kế."""

    sidebar = _config()["theme"]["sidebar"]
    assert _luminance(sidebar["backgroundColor"]) < 0.1


def test_the_sidebar_section_did_not_swallow_the_app_wide_theme_keys() -> None:
    """Bẫy của TOML: mọi khoá sau một table header thuộc về table đó. Đặt
    `[theme.sidebar]` vào giữa các khoá của `[theme]` sẽ nuốt phần còn lại —
    đã xảy ra, và `font` cùng `baseFontSize` rơi vào sidebar, tức Montserrat
    chỉ áp cho sidebar."""

    theme = _config()["theme"]
    for key in ("font", "baseFontSize", "fontFaces"):
        assert key in theme, (
            f"`{key}` không còn trong [theme] — nhiều khả năng một table header "
            "được chèn vào trước nó và nuốt mất"
        )
    assert "font" not in theme["sidebar"], (
        "sidebar khai font riêng; nếu là cố ý thì bỏ dòng này, còn không thì "
        "đây là dấu hiệu [theme.sidebar] đang nằm sai chỗ"
    )


# --------------------------------------------------------------------------
# Vùng thả file trong sidebar
#
# Lỗi thật, 2026-08-23: dòng "256MB per file · PNG, JPG, TIF" là chữ trắng trên
# nền trắng, không đọc được gì. Nguyên nhân giống hệt lần trước — một luật CSS
# **không giới hạn phạm vi** đặt nền sáng cho MỌI vùng thả file, kể cả trong
# sidebar, nơi `[theme.sidebar]` đặt chữ màu gần trắng. Nửa quyết định này ở
# CSS, nửa kia ở config, và không ai đối chiếu chúng.
# --------------------------------------------------------------------------


def _rule_body(css: str, selector: str) -> str:
    """Thân của luật có selector đúng bằng chuỗi này."""

    import re

    match = re.search(
        r"(?:^|\})\s*" + re.escape(selector) + r"\s*\{([^}]*)\}", css, re.M)
    return match.group(1) if match else ""


def test_the_sidebar_dropzone_is_dark_so_its_caption_can_be_read() -> None:
    css = STYLES_PATH.read_text(encoding="utf-8")
    body = _rule_body(
        css, '[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]')
    assert body, "không còn luật riêng cho vùng thả file trong sidebar"

    import re

    match = re.search(r"background:\s*(#[0-9a-fA-F]{6})", body)
    assert match, "luật này phải tự đặt nền, không được để luật chung áp nền sáng"

    sidebar = _config()["theme"]["sidebar"]
    ratio = _contrast(sidebar["textColor"], match.group(1))
    assert ratio >= 4.5, (
        f"chữ {sidebar['textColor']} trên nền vùng thả file {match.group(1)} chỉ "
        f"tương phản {ratio:.2f}:1 — đây đúng là lỗi 'không thấy chữ 256MB'"
    )


def test_the_light_dropzone_rule_cannot_reach_the_sidebar() -> None:
    """`stAppViewContainer` BAO CẢ sidebar, nên khoanh vùng bằng nó là vô hiệu.
    Phải là `stMain`, thứ thật sự loại sidebar ra."""

    css = STYLES_PATH.read_text(encoding="utf-8")
    assert '[data-testid="stMain"] [data-testid="stFileUploaderDropzone"]' in css
    for line in css.splitlines():
        stripped = line.strip()
        if stripped.startswith('[data-testid="stFileUploaderDropzone"]'):
            raise AssertionError(
                "luật vùng thả file không giới hạn phạm vi sẽ với tới cả sidebar: "
                f"{stripped}"
            )
