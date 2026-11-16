
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

const pages = document.querySelectorAll("[data-page]");

const settingsNavigationItems = document.querySelectorAll(
    "[data-settings-target]"
);

const settingsSections = document.querySelectorAll(
    "[data-settings-section]"
);

const topbarTitle = document.querySelector(".topbar-title");
const topbarOverline = document.querySelector(".topbar-overline");

const voiceButton = document.getElementById("voice-button");
const voiceStatus = document.getElementById("voice-status");

/* =========================================
   APPLICATION STATE
========================================= */

let bridgeReady = false;
let isProcessing = false;
let bridgeInitializationPromise = null;
let mediaRecorder = null;
let audioStream = null;
let audioChunks = [];
let isRecording = false;
let isVoiceProcessing = false;
let currentVoiceAudio = null;

/* =========================================
   PAGE NAVIGATION
========================================= */

function setActivePage(target) {
    const selectedPage = Array.from(pages).find(
        (page) => page.dataset.page === target
    );

    if (!selectedPage) {
        return false;
    }

    pages.forEach((page) => {
        const isActive = page === selectedPage;

        page.hidden = !isActive;
        page.setAttribute("aria-hidden", String(!isActive));
    });

    if (topbarOverline) {
        topbarOverline.textContent = target === "settings"
            ? "PREFERENCES"
            : "WORKSPACE";
    }

    return true;
}

function setActiveNavigation(target) {
    const selectedItem = Array.from(navigationItems).find(
        (item) => item.dataset.navTarget === target
    );

    if (!selectedItem) {
        return false;
    }

    // Validate and activate the page first.
    if (!setActivePage(target)) {
        return false;
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

    return true;
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
   SETTINGS CATEGORY NAVIGATION
========================================= */

function setActiveSettingsSection(target) {
    const selectedItem = Array.from(settingsNavigationItems).find(
        (item) => item.dataset.settingsTarget === target
    );

    const selectedSection = Array.from(settingsSections).find(
        (section) => section.dataset.settingsSection === target
    );

    if (!selectedItem || !selectedSection) {
        return;
    }

    settingsNavigationItems.forEach((item) => {
        const isActive = item === selectedItem;

        item.classList.toggle("active", isActive);

        if (isActive) {
            item.setAttribute("aria-current", "true");
        } else {
            item.removeAttribute("aria-current");
        }
    });

    settingsSections.forEach((section) => {
        const isActive = section === selectedSection;

        section.hidden = !isActive;
        section.setAttribute("aria-hidden", String(!isActive));
    });
}

function initializeSettingsNavigation() {
    settingsNavigationItems.forEach((item) => {
        item.addEventListener("click", (event) => {
            event.preventDefault();

            const target = item.dataset.settingsTarget;

            if (!target) {
                return;
            }

            setActiveSettingsSection(target);
        });
    });

    const initialActiveItem = document.querySelector(
        "[data-settings-target].active"
    );

    if (initialActiveItem) {
        setActiveSettingsSection(
            initialActiveItem.dataset.settingsTarget
        );
    }
}


/* =========================================
   MEMORY SEARCH & CATEGORY FILTERING
========================================= */

const memoryList = document.getElementById("memory-list");

const memorySearchInput = document.getElementById(
    "memory-search"
);

const memoryCategoryButtons = document.querySelectorAll(
    "[data-memory-filter]"
);

const memoryCount = document.querySelector(
    "[data-memory-count]"
);

const memoryTotalStat = document.querySelector(
    '[data-memory-stat="total"]'
);

const memoryEmptyState = document.getElementById(
    "memory-empty-state"
);

const memoryEmptyTitle = memoryEmptyState?.querySelector("h3");
const memoryEmptyDescription = memoryEmptyState?.querySelector("p");

let activeMemoryCategory = "all";
let memorySearchQuery = "";

const DEFAULT_MEMORY_EMPTY_STATE = {
    title: "NO MEMORIES FOUND",
    description:
        "Your stored memories will appear here once S7VEN starts remembering information.",
};

const FILTERED_MEMORY_EMPTY_STATE = {
    title: "NO MATCHING MEMORIES",
    description:
        "Try another search term or select a different category.",
};

/* Get all rendered memory records */

function getMemoryCards() {
    if (!memoryList) {
        return [];
    }

    return Array.from(
        memoryList.querySelectorAll(".memory-card")
    );
}

/* Update memory category navigation */

function updateMemoryCategoryNavigation() {
    memoryCategoryButtons.forEach((button) => {
        const isActive = (
            button.dataset.memoryFilter === activeMemoryCategory
        );

        button.classList.toggle("active", isActive);

        button.setAttribute(
            "aria-pressed",
            String(isActive)
        );
    });
}

/* Update empty state */

function updateMemoryEmptyState(total, visible) {
    if (!memoryEmptyState) {
        return;
    }

    const hasNoRecords = total === 0;
    const hasNoMatches = total > 0 && visible === 0;

    memoryEmptyState.hidden = !(
        hasNoRecords || hasNoMatches
    );

    if (!memoryEmptyState.hidden) {
        const emptyContent = hasNoRecords
            ? DEFAULT_MEMORY_EMPTY_STATE
            : FILTERED_MEMORY_EMPTY_STATE;

        if (memoryEmptyTitle) {
            memoryEmptyTitle.textContent = emptyContent.title;
        }

        if (memoryEmptyDescription) {
            memoryEmptyDescription.textContent = (
                emptyContent.description
            );
        }
    }
}

/* Update memory counters */

function updateMemoryCounters(total, visible) {
    if (memoryTotalStat) {
        memoryTotalStat.textContent = String(total);
    }

    if (memoryCount) {
        memoryCount.textContent = (
            `${visible} ${visible === 1 ? "RECORD" : "RECORDS"}`
        );
    }
}

/* Apply search and category filters */

function filterMemoryCards() {
    const cards = getMemoryCards();

    const normalizedQuery = memorySearchQuery
        .trim()
        .toLowerCase();

    let visibleCount = 0;

    cards.forEach((card) => {
        const category = (
            card.dataset.memoryCategory || ""
        ).toLowerCase();

        const searchableContent = (
            card.textContent || ""
        ).toLowerCase();

        const matchesCategory = (
            activeMemoryCategory === "all" ||
            category === activeMemoryCategory
        );

        const matchesSearch = (
            !normalizedQuery ||
            searchableContent.includes(normalizedQuery)
        );

        const isVisible = (
            matchesCategory &&
            matchesSearch
        );

        card.hidden = !isVisible;

        if (isVisible) {
            visibleCount += 1;
        }
    });

    updateMemoryCategoryNavigation();

    updateMemoryCounters(
        cards.length,
        visibleCount
    );

    updateMemoryEmptyState(
        cards.length,
        visibleCount
    );

    return {
        total: cards.length,
        visible: visibleCount,
    };
}

/* Initialize memory search */

function initializeMemorySearch() {
    if (!memorySearchInput) {
        return;
    }

    memorySearchInput.addEventListener("input", () => {
        memorySearchQuery = memorySearchInput.value;

        filterMemoryCards();
    });
}

/* Initialize memory category filters */

function initializeMemoryFilters() {
    memoryCategoryButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const category = button.dataset.memoryFilter;

            if (!category) {
                return;
            }

            activeMemoryCategory = category;

            filterMemoryCards();
        });
    });
}

/* Initialize memory interface */

function initializeMemoryInterface() {
    initializeMemorySearch();
    initializeMemoryFilters();

    filterMemoryCards();
}

/* Refresh after memory records are rendered */

function refreshMemoryInterface() {
    return filterMemoryCards();
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
    if (bridgeInitializationPromise) {
        return bridgeInitializationPromise;
    }

    bridgeInitializationPromise = (async () => {
        try {
            const api = window.pywebview?.api;

            if (
                !api ||
                typeof api.ping !== "function" ||
                typeof api.get_app_info !== "function"
            ) {
                throw new Error("Python bridge is unavailable.");
            }

            const result = await api.ping();

            if (!result?.success) {
                throw new Error(
                    "Python bridge connection failed."
                );
            }

            const appInfo = await api.get_app_info();

            if (!appInfo || typeof appInfo.name !== "string") {
                throw new Error(
                    "Invalid application information received."
                );
            }

            bridgeReady = true;

            updateSystemStatus(`${appInfo.name} CONNECTED`);
            updateComposer();

            console.info("S7VEN Python bridge connected.");

            return true;
        } catch (error) {
            bridgeReady = false;

            updateSystemStatus("BRIDGE UNAVAILABLE");
            updateComposer();

            console.error(
                "Bridge initialization failed:",
                error
            );

            return false;
        } finally {
            bridgeInitializationPromise = null;
        }
    })();

    return bridgeInitializationPromise;
}


/* =========================================
   VOICE RECORDING
========================================= */

function updateVoiceStatus(status) {
    if (!voiceStatus) {
        return;
    }

    const textNode = Array.from(voiceStatus.childNodes).find(
        (node) => node.nodeType === Node.TEXT_NODE &&
            node.textContent.trim()
    );

    if (textNode) {
        textNode.textContent = ` ${status}`;
    }
}

function updateVoiceButton(recording) {
    if (!voiceButton) {
        return;
    }

    voiceButton.classList.toggle("is-recording", recording);
    voiceButton.setAttribute("aria-pressed", String(recording));
    voiceButton.setAttribute(
        "aria-label",
        recording ? "Stop voice recording" : "Start voice recording"
    );

    voiceButton.title = recording
        ? "Stop recording"
        : "Push to talk";
}


async function startVoiceRecording() {
    if (!bridgeReady) {
        updateVoiceStatus("Bridge unavailable");
        console.error("Voice recording blocked: Python bridge is not ready.");
        return;
    }

    if (isProcessing || isVoiceProcessing || isRecording) {
        return;
    }

    try {
        updateVoiceStatus("Checking voice settings...");

        const api = window.pywebview?.api;

        if (!api || typeof api.get_voice_settings !== "function") {
            throw new Error("Voice settings API is unavailable.");
        }

        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("Audio recording is not supported.");
        }

        if (!window.MediaRecorder) {
            throw new Error("MediaRecorder is not supported.");
        }

        const settings = await api.get_voice_settings();

        if (!settings?.success) {
            throw new Error(settings?.error || "Unable to load voice settings.");
        }

        if (!settings.enabled || !settings.input_enabled) {
            throw new Error("Voice input is disabled in settings.");
        }

        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: true,
        });

        const mimeType = MediaRecorder.isTypeSupported(
            "audio/webm;codecs=opus"
        )
            ? "audio/webm;codecs=opus"
            : "audio/webm";

        mediaRecorder = new MediaRecorder(audioStream, { mimeType });
        audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        });

        mediaRecorder.addEventListener(
            "stop",
            handleVoiceRecordingStop,
            { once: true }
        );

        mediaRecorder.start();

        isRecording = true;
        updateVoiceButton(true);
        updateVoiceStatus("Listening...");
        updateSystemStatus("VOICE RECORDING");

    } catch (error) {
        stopAudioStream();
        updateVoiceButton(false);
        updateVoiceStatus(error.message || "Voice error");
        updateSystemStatus("VOICE ERROR");

        console.error("Voice recording failed:", error);
    }
}

function stopAudioStream() {
    if (audioStream) {
        audioStream.getTracks().forEach((track) => track.stop());
        audioStream = null;
    }
}

function stopVoiceRecording() {
    if (!mediaRecorder || mediaRecorder.state === "inactive") {
        return;
    }

    isRecording = false;
    updateVoiceButton(false);
    updateVoiceStatus("Processing audio...");

    mediaRecorder.stop();
    stopAudioStream();
}

async function playVoiceAudio(base64Audio, audioFormat = "mp3") {
    if (!base64Audio) {
        return;
    }

    const mimeTypes = {
        mp3: "audio/mpeg",
        wav: "audio/wav",
        webm: "audio/webm",
    };

    const mimeType = mimeTypes[audioFormat];

    if (!mimeType) {
        throw new Error("Unsupported audio format.");
    }

    const binary = atob(base64Audio);
    const bytes = new Uint8Array(binary.length);

    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
    }

    const audioBlob = new Blob([bytes], {
        type: mimeType,
    });

    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    currentVoiceAudio = audio;

    const playbackFinished = new Promise((resolve, reject) => {
        audio.addEventListener("ended", resolve, { once: true });

        audio.addEventListener(
            "error",
            () => reject(new Error("TTS playback failed.")),
            { once: true }
        );
    });

    try {
        await audio.play();
        await playbackFinished;
    } finally {
        if (currentVoiceAudio === audio) {
            currentVoiceAudio = null;
        }

        URL.revokeObjectURL(audioUrl);
    }
}
async function handleVoiceRecordingStop() {
    const recorder = mediaRecorder;
    mediaRecorder = null;

    if (!audioChunks.length || !recorder) {
        updateVoiceStatus("Voice standby");
        return;
    }

    isVoiceProcessing = true;
    updateVoiceStatus("Processing audio...");
    updateSystemStatus("PROCESSING VOICE...");

    try {
        const audioBlob = new Blob(audioChunks, {
            type: recorder.mimeType || "audio/webm"
        });

        audioChunks = [];

        const arrayBuffer = await audioBlob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);

        let binary = "";

        bytes.forEach((byte) => {
            binary += String.fromCharCode(byte);
        });

        const payload = btoa(binary);

        const result = await window.pywebview.api.process_audio(
            payload,
            "webm",
            24000,
            null
        );

        if (!result?.success) {
            throw new Error(
                result?.error || "Voice processing failed."
            );
        }

        if (result.transcription) {
            appendMessage(result.transcription, "user");
        }

        if (result.response) {
            appendMessage(result.response, "assistant");
        }

        if (result.audio) {
            updateVoiceStatus("Speaking...");
            await playVoiceAudio(
                result.audio,
                result.audio_format
            );
        }

        updateVoiceStatus("Voice standby");
    } catch (error) {
        updateVoiceStatus("Voice error");
        console.error("Voice processing failed:", error);
    } finally {
        isVoiceProcessing = false;
        updateSystemStatus("SYSTEM READY");
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

        if (
            messageForm &&
            sendButton &&
            !sendButton.disabled
        ) {
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

if (voiceButton) {
    voiceButton.addEventListener("click", () => {
        if (isRecording) {
            stopVoiceRecording();
        } else {
            startVoiceRecording();
        }
    });
}

/* =========================================
   INITIALIZATION
========================================= */

initializeNavigation();
initializeSettingsNavigation();
initializeMemoryInterface();

window.addEventListener(
    "pywebviewready",
    initializeBridge,
    { once: true }
);

// Handle cases where the bridge is already available.
if (window.pywebview?.api) {
    initializeBridge();
}

updateComposer();