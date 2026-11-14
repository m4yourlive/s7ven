
from pathlib import Path


FRONTEND_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "seven"
    / "gui"
    / "frontend"
)

JS_FILE = FRONTEND_DIR / "assets" / "js" / "app.js"


def read_js() -> str:
    return JS_FILE.read_text(encoding="utf-8")

HTML_FILE = FRONTEND_DIR / "index.html"
CSS_FILE = FRONTEND_DIR / "assets" / "css" / "style.css"


def read_html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


def read_css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def test_memory_navigation_exists():
    html = read_html()

    assert 'data-nav-target="memory"' in html


def test_memory_page_exists():
    html = read_html()

    assert 'id="memory-page"' in html
    assert 'data-page="memory"' in html


def test_memory_page_starts_hidden():
    html = read_html()

    memory_page = html.split(
        'id="memory-page"',
        maxsplit=1,
    )[1].split("</section>", maxsplit=1)[0]

    assert "hidden" in memory_page


def test_memory_header_exists():
    html = read_html()

    assert 'class="memory-header"' in html
    assert 'class="memory-title"' in html


def test_memory_statistics_exist():
    html = read_html()

    assert 'class="memory-stats"' in html
    assert 'data-memory-stat="total"' in html
    assert 'data-memory-stat="categories"' in html
    assert 'data-memory-stat="storage"' in html


def test_memory_search_exists():
    html = read_html()

    assert 'id="memory-search"' in html
    assert "data-memory-search" in html


def test_memory_categories_exist():
    html = read_html()

    categories = [
        "all",
        "personal",
        "preferences",
        "projects",
        "knowledge",
    ]

    for category in categories:
        assert f'data-memory-filter="{category}"' in html


def test_memory_list_exists():
    html = read_html()

    assert 'id="memory-list"' in html
    assert 'class="memory-list"' in html


def test_memory_empty_state_exists():
    html = read_html()

    assert 'id="memory-empty-state"' in html
    assert "NO MEMORIES FOUND" in html


def test_memory_accessibility_labels_exist():
    html = read_html()

    assert 'aria-label="Memory management"' in html
    assert 'aria-label="Memory categories"' in html
    assert 'aria-label="Stored memories"' in html


def test_memory_styles_exist():
    css = read_css()

    required_classes = [
        ".memory-page",
        ".memory-header",
        ".memory-title",
        ".memory-stats",
        ".memory-stat-card",
        ".memory-toolbar",
        ".memory-categories",
        ".memory-category",
        ".memory-list",
        ".memory-empty-state",
    ]

    for selector in required_classes:
        assert selector in css


def test_memory_responsive_layout_exists():
    css = read_css()

    assert "@media (max-width: 760px)" in css
    assert "@media (max-width: 560px)" in css



def test_memory_browser_layout_exists():
    html = read_html()

    assert 'class="memory-browser"' in html
    assert 'class="memory-list-section"' in html
    assert 'class="memory-detail-panel"' in html


def test_memory_detail_panel_has_empty_state():
    html = read_html()

    assert 'aria-label="Memory details"' in html
    assert 'class="memory-detail-empty"' in html
    assert "NO MEMORY SELECTED" in html
    assert "AWAITING SELECTION" in html


def test_memory_detail_panel_has_status_indicator():
    html = read_html()

    assert 'class="memory-detail-indicator"' in html
    assert "STANDBY" in html


def test_memory_card_component_styles_exist():
    css = read_css()

    required_selectors = [
        ".memory-card",
        ".memory-card-header",
        ".memory-card-category",
        ".memory-card-title",
        ".memory-card-content",
        ".memory-card-footer",
        ".memory-card-action",
        ".memory-card.is-selected",
    ]

    for selector in required_selectors:
        assert selector in css


def test_memory_detail_component_styles_exist():
    css = read_css()

    required_selectors = [
        ".memory-detail-panel",
        ".memory-detail-header",
        ".memory-detail-indicator",
        ".memory-detail-empty",
        ".memory-detail-icon",
        ".memory-detail-footer",
    ]

    for selector in required_selectors:
        assert selector in css


def test_memory_browser_responsive_layout():
    css = read_css()

    assert "grid-template-columns: minmax(0, 1fr) 290px" in css
    assert "@media (max-width: 900px)" in css
    assert "@media (max-width: 760px)" in css


def test_memory_keyboard_focus_styles_exist():
    css = read_css()

    assert ".memory-category:focus-visible" in css
    assert ".memory-search-wrapper input:focus-visible" in css


def test_memory_search_elements_exist():
    html = read_html()

    assert 'id="memory-search"' in html
    assert "data-memory-search" in html


def test_memory_category_buttons_have_accessibility_state():
    html = read_html()

    categories = [
        "all",
        "personal",
        "preferences",
        "projects",
        "knowledge",
    ]

    for category in categories:
        assert f'data-memory-filter="{category}"' in html

    assert 'aria-pressed="true"' in html
    assert 'aria-pressed="false"' in html


def test_memory_search_logic_exists():
    js = read_js()

    required_functions = [
        "initializeMemorySearch",
        "filterMemoryCards",
        "getMemoryCards",
        "refreshMemoryInterface",
    ]

    for function in required_functions:
        assert f"function {function}" in js


def test_memory_category_filter_logic_exists():
    js = read_js()

    assert "initializeMemoryFilters" in js
    assert "activeMemoryCategory" in js
    assert "dataset.memoryCategory" in js
    assert "matchesCategory" in js


def test_memory_search_supports_case_insensitive_matching():
    js = read_js()

    assert ".toLowerCase()" in js
    assert "includes(normalizedQuery)" in js


def test_memory_filter_updates_card_visibility():
    js = read_js()

    assert "card.hidden = !isVisible" in js
    assert "visibleCount += 1" in js


def test_memory_empty_state_supports_search_results():
    js = read_js()

    assert "NO MEMORIES FOUND" in js
    assert "NO MATCHING MEMORIES" in js
    assert "updateMemoryEmptyState" in js


def test_memory_counters_are_updated():
    js = read_js()

    assert "updateMemoryCounters" in js
    assert "data-memory-stat" in js
    assert "data-memory-count" in js


def test_memory_interface_is_initialized():
    js = read_js()

    assert "initializeMemoryInterface();" in js


def test_memory_refresh_function_is_available():
    js = read_js()

    assert "function refreshMemoryInterface()" in js
    assert "return filterMemoryCards();" in js


def test_memory_hidden_elements_respect_hidden_attribute():
    css = read_css()

    assert ".memory-card[hidden]" in css
    assert ".memory-empty-state[hidden]" in css
    assert "display: none !important" in css
