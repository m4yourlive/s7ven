from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_ROOT = (
    PROJECT_ROOT
    / "src"
    / "seven"
    / "gui"
    / "frontend"
)

FRONTEND_ENTRY = FRONTEND_ROOT / "index.html"


class AssetReferenceParser(HTMLParser):
    """Collect local asset references from HTML elements."""

    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        for attribute, value in attrs:
            if attribute not in {"src", "href"}:
                continue

            if value is None or not value.strip():
                continue

            self.references.append(value.strip())


def test_frontend_directory_exists() -> None:
    assert FRONTEND_ROOT.is_dir()


def test_frontend_entry_point_exists() -> None:
    assert FRONTEND_ENTRY.is_file()


def test_frontend_entry_point_is_readable() -> None:
    content = FRONTEND_ENTRY.read_text(encoding="utf-8")

    assert content.strip()


def test_frontend_entry_point_has_valid_html_structure() -> None:
    content = FRONTEND_ENTRY.read_text(encoding="utf-8")
    normalized = content.lower()

    assert "<!doctype html" in normalized
    assert "<html" in normalized
    assert "</html>" in normalized
    assert "<head" in normalized
    assert "</head>" in normalized
    assert "<body" in normalized
    assert "</body>" in normalized


def test_frontend_local_assets_exist() -> None:
    content = FRONTEND_ENTRY.read_text(encoding="utf-8")

    parser = AssetReferenceParser()
    parser.feed(content)
    parser.close()

    missing_assets: list[str] = []

    for reference in parser.references:
        parsed = urlsplit(reference)

        if parsed.scheme or parsed.netloc:
            continue

        asset_path = unquote(parsed.path)

        if not asset_path:
            continue

        if asset_path.startswith("/"):
            resolved_path = FRONTEND_ROOT / asset_path.lstrip("/")
        else:
            resolved_path = FRONTEND_ROOT / asset_path

        if not resolved_path.is_file():
            missing_assets.append(reference)

    assert not missing_assets, (
        "Missing frontend assets:\n"
        + "\n".join(missing_assets)
    )
