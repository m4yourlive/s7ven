
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = (
    PROJECT_ROOT
    / "src"
    / "seven"
    / "gui"
    / "frontend"
)

HTML_FILE = FRONTEND_DIR / "index.html"
CSS_FILE = FRONTEND_DIR / "assets" / "css" / "style.css"
JS_FILE = FRONTEND_DIR / "assets" / "js" / "app.js"


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_navigation_item(html: str, target: str) -> str:
    """Return the complete anchor element for a navigation target."""
    marker = f'data-nav-target="{target}"'

    marker_position = html.index(marker)

    opening_tag_start = html.rfind("<a", 0, marker_position)
    opening_tag_end = html.index(">", marker_position)

    closing_tag_end = html.index("</a>", opening_tag_end)

    return html[opening_tag_start:closing_tag_end + 4]


def test_navigation_elements_exist() -> None:
    html = read_file(HTML_FILE)

    assert 'class="navigation"' in html
    assert 'aria-label="Main navigation"' in html


def test_navigation_contains_all_items() -> None:
    html = read_file(HTML_FILE)

    assert 'data-nav-target="new-session"' in html
    assert 'data-nav-target="conversations"' in html
    assert 'data-nav-target="memory"' in html


def test_navigation_has_initial_active_state() -> None:
    html = read_file(HTML_FILE)

    navigation_item = get_navigation_item(
        html,
        "new-session",
    )

    assert 'class="nav-item active"' in navigation_item
    assert 'aria-current="page"' in navigation_item


def test_other_navigation_items_are_not_active() -> None:
    html = read_file(HTML_FILE)

    for target in ("conversations", "memory"):
        navigation_item = get_navigation_item(html, target)

        assert 'class="nav-item active"' not in navigation_item
        assert 'aria-current="page"' not in navigation_item


def test_navigation_contains_section_headings() -> None:
    html = read_file(HTML_FILE)

    assert "WORKSPACE" in html
    assert "PERSONAL" in html


def test_navigation_javascript_initializes() -> None:
    js = read_file(JS_FILE)

    assert "function initializeNavigation()" in js
    assert "initializeNavigation();" in js


def test_navigation_supports_active_state() -> None:
    js = read_file(JS_FILE)

    assert "function setActiveNavigation(target)" in js
    assert 'classList.toggle("active", isActive)' in js
    assert 'setAttribute("aria-current", "page")' in js
    assert 'removeAttribute("aria-current")' in js


def test_navigation_updates_topbar_title() -> None:
    js = read_file(JS_FILE)

    assert 'document.querySelector(".topbar-title")' in js
    assert "topbarTitle.textContent" in js


def test_navigation_css_contains_interaction_states() -> None:
    css = read_file(CSS_FILE)

    assert ".nav-item:hover" in css
    assert ".nav-item.active" in css
    assert ".nav-item:focus-visible" in css


def test_navigation_preserves_sidebar_status() -> None:
    html = read_file(HTML_FILE)
    css = read_file(CSS_FILE)

    assert 'class="sidebar-bottom"' in html
    assert 'class="system-status"' in html
    assert ".system-status" in css


def test_navigation_preserves_python_bridge() -> None:
    js = read_file(JS_FILE)

    assert "window.pywebview?.api" in js
    assert "api.ping()" in js
    assert "api.get_app_info()" in js
    assert "api.send_message(" in js
