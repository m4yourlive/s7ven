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


def test_conversation_container_exists() -> None:
    html = read_file(HTML_FILE)

    assert 'id="conversation"' in html
    assert 'class="conversation-area"' in html
    assert 'aria-live="polite"' in html


def test_conversation_has_empty_state() -> None:
    html = read_file(HTML_FILE)

    assert 'class="empty-state"' in html
    assert 'class="empty-state-mark"' in html


def test_conversation_has_message_renderer() -> None:
    js = read_file(JS_FILE)

    assert "function appendMessage(content, sender)" in js
    assert "document.createElement(" in js
    assert "conversation.appendChild(messageElement)" in js


def test_empty_state_is_removed_when_message_arrives() -> None:
    js = read_file(JS_FILE)

    assert "function removeEmptyState()" in js
    assert 'querySelector(".empty-state")' in js
    assert "emptyState.remove()" in js
    assert "removeEmptyState();" in js


def test_messages_support_user_and_assistant_senders() -> None:
    js = read_file(JS_FILE)

    assert 'sender === "user"' in js
    assert 'message-${normalizedSender}' in js
    assert 'normalizedSender === "user"' in js


def test_message_content_uses_text_content() -> None:
    js = read_file(JS_FILE)

    assert "contentElement.textContent = String(content)" in js
    assert "contentElement.innerHTML" not in js


def test_message_renderer_returns_message_element() -> None:
    js = read_file(JS_FILE)

    assert "return messageElement;" in js


def test_conversation_scrolls_to_latest_message() -> None:
    js = read_file(JS_FILE)

    assert "function scrollConversationToBottom()" in js
    assert "conversation.scrollTop = conversation.scrollHeight" in js
    assert "scrollConversationToBottom();" in js


def test_message_submission_uses_conversation_renderer() -> None:
    js = read_file(JS_FILE)

    assert "async function handleMessageSubmit(event)" in js
    assert 'appendMessage(message, "user")' in js
    assert 'appendMessage(result.reply, "assistant")' in js


def test_message_submission_handles_errors() -> None:
    js = read_file(JS_FILE)

    assert "catch (error)" in js
    assert "Unable to communicate with S7VEN Core." in js


def test_conversation_css_contains_message_styles() -> None:
    css = read_file(CSS_FILE)

    assert ".conversation-area" in css
    assert ".conversation-has-messages" in css
    assert ".message" in css
    assert ".message-user" in css
    assert ".message-assistant" in css
    assert ".message-content" in css


def test_conversation_css_contains_responsive_styles() -> None:
    css = read_file(CSS_FILE)

    assert "@media (max-width: 760px)" in css
    assert "@media (max-width: 560px)" in css


def test_conversation_css_contains_message_animation() -> None:
    css = read_file(CSS_FILE)

    assert "@keyframes message-appear" in css
    assert "message-appear 180ms ease-out" in css


def test_python_bridge_is_preserved() -> None:
    js = read_file(JS_FILE)

    assert "window.pywebview?.api" in js
    assert "api.ping()" in js
    assert "api.get_app_info()" in js
    assert "window.pywebview.api.send_message(" in js


def test_navigation_is_preserved() -> None:
    js = read_file(JS_FILE)

    assert "function initializeNavigation()" in js
    assert "initializeNavigation();" in js
    assert "function setActiveNavigation(target)" in js
