
"use strict";

/* =========================================
   CONSTANTS
========================================= */

const MAX_MESSAGE_LENGTH = 4000;
const MAX_COMPOSER_HEIGHT = 150;
const CHARACTER_WARNING_THRESHOLD = 0.9;

/* =========================================
   DOM ELEMENTS
========================================= */

const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messageForm = document.getElementById("message-form");
const conversation = document.getElementById("conversation");
const systemStatus = document.querySelector(".system-status");

const characterCounter = document.getElementById(
    "character-counter"
);

const navigationItems = document.querySelectorAll(
    "[data-nav-target]"
);

const topbarTitle = document.querySelector(".topbar-title");

/* =========================================
   APPLICATION STATE
========================================= */

let bridgeReady = false;
let isProcessing = false;

/* =========================================
   NAVIGATION
========================================= */

function setActiveNavigation(target) {
    const selectedItem = Array.from(navigationItems).find(
        (item) => item.dataset.navTarget === target
    );

    if (!selectedItem) {
        return;
    }

    navigationItems.forEach((item) => {
        const isActive = item === selectedItem;

        item.classList.toggle("active", isActive);

        if (isActive) {
            item.setAttribute("aria-current", "page");
        } else {
            item.removeAttribute("aria-current");
        }
    });

    if (topbarTitle) {
        const label = selectedItem.querySelector(".nav-label");

        if (label) {
            topbarTitle.textContent = label.textContent.trim();
        }
    }
}

function initializeNavigation() {
    navigationItems.forEach((item) => {
        item.addEventListener("click", (event) => {
            event.preventDefault();

            const target = item.dataset.navTarget;

            if (!target) {
                return;
            }

            setActiveNavigation(target);
        });
    });

    const initialActiveItem = document.querySelector(
        "[data-nav-target].active"
    );

    if (initialActiveItem) {
        setActiveNavigation(initialActiveItem.dataset.navTarget);
    }
}

/* =========================================
   SYSTEM STATUS
========================================= */

function updateSystemStatus(status) {
    if (!systemStatus) {
        return;
    }

    const statusText = systemStatus.lastElementChild;

    if (statusText) {
        statusText.textContent = status;
    }
}

/* =========================================
   MESSAGE COMPOSER
========================================= */

function updateCharacterCounter() {
    if (!messageInput || !characterCounter) {
        return;
    }

    const length = messageInput.value.length;
    const percentage = length / MAX_MESSAGE_LENGTH;

    characterCounter.textContent = (
        `${length} / ${MAX_MESSAGE_LENGTH}`
    );

    characterCounter.classList.toggle(
        "near-limit",
        percentage >= CHARACTER_WARNING_THRESHOLD
    );
}

function resizeMessageInput() {
    if (!messageInput) {
        return;
    }

    messageInput.style.height = "auto";

    const nextHeight = Math.min(
        messageInput.scrollHeight,
        MAX_COMPOSER_HEIGHT
    );

    messageInput.style.height = `${nextHeight}px`;
}

function updateComposer() {
    if (!messageInput || !sendButton) {
        return;
    }

    const message = messageInput.value;
    const hasMessage = message.trim().length > 0;

    const isWithinLimit = (
        message.length <= MAX_MESSAGE_LENGTH
    );

    const canSend = (
        hasMessage &&
        isWithinLimit &&
        bridgeReady &&
        !isProcessing
    );

    sendButton.disabled = !canSend;

    updateCharacterCounter();
    resizeMessageInput();
}

function setComposerProcessing(processing) {
    isProcessing = processing;

    if (!messageInput || !sendButton) {
        return;
    }

    messageInput.disabled = processing;

    sendButton.classList.toggle(
        "is-loading",
        processing
    );

    sendButton.setAttribute(
        "aria-busy",
        String(processing)
    );

    sendButton.setAttribute(
        "aria-label",
        processing
            ? "S7VEN is responding"
            : "Send message"
    );

    const icon = sendButton.querySelector(".send-button-icon");

    if (icon) {
        icon.textContent = processing ? "..." : "↑";
    }

    updateComposer();
}

/* =========================================
   CONVERSATION INTERFACE
========================================= */

function removeEmptyState() {
    if (!conversation) {
        return;
    }

    const emptyState = conversation.querySelector(".empty-state");

    if (emptyState) {
        emptyState.remove();
    }

    conversation.classList.add("conversation-has-messages");
}

function scrollConversationToBottom() {
    if (!conversation) {
        return;
    }

    conversation.scrollTop = conversation.scrollHeight;
}

function appendMessage(content, sender) {
    if (!conversation) {
        return null;
    }

    const normalizedSender = sender === "user"
        ? "user"
        : "assistant";

    removeEmptyState();

    const messageElement = document.createElement("article");
    const labelElement = document.createElement("span");
    const contentElement = document.createElement("p");

    messageElement.className = (
        `message message-${normalizedSender}`
    );

    messageElement.dataset.sender = normalizedSender;

    messageElement.setAttribute(
        "aria-label",
        normalizedSender === "user"
            ? "Your message"
            : "S7VEN response"
    );

    labelElement.className = "message-label";

    labelElement.textContent = normalizedSender === "user"
        ? "YOU"
        : "S7VEN";

    contentElement.className = "message-content";
    contentElement.textContent = String(content);

    messageElement.append(labelElement, contentElement);
    conversation.appendChild(messageElement);

    scrollConversationToBottom();

    return messageElement;
}

/* =========================================
   PYTHON BRIDGE
========================================= */

async function initializeBridge() {
    try {
        const api = window.pywebview?.api;

        if (!api) {
            throw new Error("Python bridge is unavailable.");
        }

        const result = await api.ping();

        if (!result.success) {
            throw new Error("Python bridge connection failed.");
        }

        const appInfo = await api.get_app_info();

        bridgeReady = true;

        updateSystemStatus(`${appInfo.name} CONNECTED`);
        updateComposer();

        console.info("S7VEN Python bridge connected.");
    } catch (error) {
        bridgeReady = false;

        updateSystemStatus("BRIDGE UNAVAILABLE");
        updateComposer();

        console.error("Bridge initialization failed:", error);
    }
}

/* =========================================
   MESSAGE SUBMISSION
========================================= */

async function handleMessageSubmit(event) {
    event.preventDefault();

    if (
        !messageInput ||
        !sendButton ||
        !bridgeReady ||
        isProcessing
    ) {
        return;
    }

    const message = messageInput.value.trim();

    if (
        !message ||
        message.length > MAX_MESSAGE_LENGTH
    ) {
        return;
    }

    messageInput.value = "";

    updateComposer();
    appendMessage(message, "user");

    setComposerProcessing(true);
    updateSystemStatus("S7VEN IS THINKING...");

    try {
        const result = await window.pywebview.api.send_message(
            message
        );

        if (!result.success) {
            throw new Error(
                result.error || "Message submission failed."
            );
        }

        appendMessage(result.reply, "assistant");
    } catch (error) {
        appendMessage(
            "Unable to communicate with S7VEN Core.",
            "assistant"
        );

        console.error("Message bridge error:", error);
    } finally {
        setComposerProcessing(false);

        updateSystemStatus(
            bridgeReady
                ? "SYSTEM READY"
                : "BRIDGE UNAVAILABLE"
        );

        updateComposer();
        messageInput.focus();
    }
}

/* =========================================
   EVENT LISTENERS
========================================= */

if (messageInput) {
    messageInput.addEventListener("input", updateComposer);

    messageInput.addEventListener("keydown", (event) => {
        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.isComposing
        ) {
            event.preventDefault();

            if (sendButton && !sendButton.disabled) {
                messageForm.requestSubmit();
            }
        }
    });
}

if (messageForm) {
    messageForm.addEventListener(
        "submit",
        handleMessageSubmit
    );
}

/* =========================================
   INITIALIZATION
========================================= */

initializeNavigation();

window.addEventListener(
    "pywebviewready",
    initializeBridge
);

updateComposer();
