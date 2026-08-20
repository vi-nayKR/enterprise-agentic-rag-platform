document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const userPromptInput = document.getElementById("user-prompt-input");
    const submitBtn = document.getElementById("submit-btn");
    const chatInner = document.getElementById("chat-inner");
    const welcomeScreen = document.getElementById("welcome-screen");
    const chatScrollContainer = document.getElementById("chat-scroll-container");
    const newChatBtn = document.getElementById("new-chat-btn");
    const toggleSidebarBtn = document.getElementById("toggle-sidebar-btn");
    const mobileMenuBtn = document.getElementById("mobile-menu-btn");
    const sidebar = document.getElementById("sidebar");
    const fileUploadInput = document.getElementById("file-upload-input");
    const uploadProgressBar = document.getElementById("upload-progress-bar");
    const uploadStatusText = document.getElementById("upload-status-text");
    const documentsList = document.getElementById("documents-list");
    const mcpToolsList = document.getElementById("mcp-tools-list");
    const toolsCountBadge = document.getElementById("tools-count-badge");
    const citationDrawer = document.getElementById("citation-drawer");
    const drawerContent = document.getElementById("drawer-content");
    const closeDrawerBtn = document.getElementById("close-drawer-btn");
    const dragOverlay = document.getElementById("drag-overlay");

    let isGenerating = false;

    // 1. Initial Data Fetching
    loadDocuments();
    loadMCPTools();

    // 2. Sidebar Toggle
    if (toggleSidebarBtn) {
        toggleSidebarBtn.addEventListener("click", () => {
            sidebar.classList.toggle("collapsed");
        });
    }
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener("click", () => {
            sidebar.classList.toggle("collapsed");
        });
    }

    // 3. Auto-growing Textarea & Submit Button State
    userPromptInput.addEventListener("input", () => {
        userPromptInput.style.height = "auto";
        userPromptInput.style.height = `${Math.min(userPromptInput.scrollHeight, 160)}px`;
        submitBtn.disabled = !userPromptInput.value.trim() || isGenerating;
    });

    userPromptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!submitBtn.disabled) {
                chatForm.dispatchEvent(new Event("submit"));
            }
        }
    });

    // 4. New Chat
    newChatBtn.addEventListener("click", () => {
        chatInner.innerHTML = "";
        if (welcomeScreen) {
            chatInner.appendChild(welcomeScreen);
        }
        userPromptInput.value = "";
        userPromptInput.style.height = "auto";
        submitBtn.disabled = true;
        closeCitationDrawer();
        if (window.lucide) lucide.createIcons();
    });

    // 5. Submit Query & Stream Response
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = userPromptInput.value.trim();
        if (!query || isGenerating) return;

        // Hide welcome screen
        if (welcomeScreen && welcomeScreen.parentElement) {
            welcomeScreen.remove();
        }

        userPromptInput.value = "";
        userPromptInput.style.height = "auto";
        submitBtn.disabled = true;
        isGenerating = true;

        // Render User Message
        appendUserMessage(query);

        // Render Assistant Placeholder
        const assistantElement = createAssistantMessageElement();
        chatInner.appendChild(assistantElement);
        scrollToBottom();

        const thoughtDropdown = assistantElement.querySelector(".thought-dropdown");
        const thoughtDetails = assistantElement.querySelector(".thought-details");
        const thoughtLabel = assistantElement.querySelector(".thought-label-text");
        const markdownBody = assistantElement.querySelector(".markdown-content");
        const citationBadges = assistantElement.querySelector(".citation-badges-wrapper");
        const cursor = assistantElement.querySelector(".streaming-cursor");

        let accumulatedText = "";
        let thoughtCount = 0;

        try {
            const response = await fetch("/api/v1/query/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, session_id: "chatgpt-session" })
            });

            if (!response.ok) {
                cursor.remove();
                markdownBody.innerHTML = `<p style="color: #ef4444;">Error: Received status ${response.status} from API.</p>`;
                isGenerating = false;
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.trim()) continue;
                    const eventMatch = line.match(/^event:\s*(\w+)/);
                    const dataMatch = line.match(/data:\s*(.+)$/m);

                    if (eventMatch && dataMatch) {
                        const eventType = eventMatch[1];
                        const data = JSON.parse(dataMatch[1]);

                        if (eventType === "thought") {
                            thoughtCount++;
                            thoughtDropdown.style.display = "block";
                            thoughtLabel.textContent = `Thinking (${thoughtCount} steps)...`;
                            const stepP = document.createElement("div");
                            stepP.textContent = `• ${data.thought}`;
                            thoughtDetails.appendChild(stepP);
                        } else if (eventType === "token") {
                            accumulatedText += data.delta;
                            if (window.marked) {
                                markdownBody.innerHTML = marked.parse(accumulatedText);
                            } else {
                                markdownBody.textContent = accumulatedText;
                            }
                            markdownBody.appendChild(cursor);
                        } else if (eventType === "citation") {
                            citationBadges.style.display = "flex";
                            const chip = document.createElement("button");
                            chip.className = "citation-chip";
                            chip.innerHTML = `<i data-lucide="file-text"></i> ${data.filename}`;
                            chip.onclick = () => openCitationDrawer(data);
                            citationBadges.appendChild(chip);
                            if (window.lucide) lucide.createIcons();
                        } else if (eventType === "done") {
                            thoughtLabel.textContent = `Thought process completed (${thoughtCount} steps)`;
                        }
                        scrollToBottom();
                    }
                }
            }

            // Remove streaming cursor
            cursor.remove();
        } catch (err) {
            cursor.remove();
            markdownBody.innerHTML = `<p style="color: #ef4444;">Connection failed: ${err.message}</p>`;
        } finally {
            isGenerating = false;
            submitBtn.disabled = !userPromptInput.value.trim();
            if (window.lucide) lucide.createIcons();
        }
    });

    // 6. Helper Renderers
    function appendUserMessage(text) {
        const row = document.createElement("div");
        row.className = "chat-message-row user";
        row.innerHTML = `
            <div class="user-bubble">${escapeHTML(text)}</div>
        `;
        chatInner.appendChild(row);
    }

    function createAssistantMessageElement() {
        const row = document.createElement("div");
        row.className = "chat-message-row assistant";
        row.innerHTML = `
            <div class="message-avatar assistant-avatar">
                <i data-lucide="sparkles"></i>
            </div>
            <div class="assistant-body">
                <div class="thought-dropdown" style="display: none;">
                    <div class="thought-toggle" onclick="this.nextElementSibling.classList.toggle('hidden')">
                        <div class="thought-label">
                            <span class="thought-pulse"></span>
                            <span class="thought-label-text">Thinking...</span>
                        </div>
                        <i data-lucide="chevron-down" style="width: 14px; height: 14px;"></i>
                    </div>
                    <div class="thought-details hidden"></div>
                </div>
                <div class="markdown-content">
                    <span class="streaming-cursor"></span>
                </div>
                <div class="citation-badges-wrapper" style="display: none;"></div>
            </div>
        `;
        if (window.lucide) lucide.createIcons();
        return row;
    }

    function scrollToBottom() {
        chatScrollContainer.scrollTop = chatScrollContainer.scrollHeight;
    }

    // 7. Citation Inspector Drawer
    function openCitationDrawer(citation) {
        drawerContent.innerHTML = `
            <div class="citation-card">
                <div class="citation-header">
                    <span class="citation-file">${escapeHTML(citation.filename || "Source Document")}</span>
                    <span class="citation-score">Score: ${citation.score ? citation.score.toFixed(4) : "0.9500"}</span>
                </div>
                <div style="font-size: 12px; color: #38bdf8;"><strong>Section:</strong> ${escapeHTML(citation.section || "General")}</div>
                <div class="citation-snippet">${escapeHTML(citation.snippet || "No text preview available.")}</div>
            </div>
        `;
        citationDrawer.classList.add("open");
    }

    function closeCitationDrawer() {
        citationDrawer.classList.remove("open");
    }

    if (closeDrawerBtn) {
        closeDrawerBtn.addEventListener("click", closeCitationDrawer);
    }

    // 8. Document Ingestion
    fileUploadInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadFile(e.target.files[0]);
        }
    });

    async function uploadFile(file) {
        uploadProgressBar.classList.remove("hidden");
        uploadStatusText.textContent = `Ingesting ${file.name}...`;

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/v1/documents/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (res.ok) {
                uploadStatusText.textContent = `Indexed ${file.name}!`;
                loadDocuments();
                setTimeout(() => uploadProgressBar.classList.add("hidden"), 2500);
            } else {
                uploadStatusText.textContent = `Failed: ${data.detail || "Upload error"}`;
            }
        } catch (e) {
            uploadStatusText.textContent = `Network error during upload.`;
        }
    }

    async function loadDocuments() {
        try {
            const res = await fetch("/api/v1/documents");
            const data = await res.json();
            documentsList.innerHTML = "";
            if (data.documents && data.documents.length > 0) {
                data.documents.forEach(doc => {
                    const li = document.createElement("li");
                    li.innerHTML = `<i data-lucide="file-text" style="width: 14px; height: 14px; color: #38bdf8;"></i> <span>${escapeHTML(doc.filename)}</span>`;
                    documentsList.appendChild(li);
                });
            } else {
                documentsList.innerHTML = `<li class="empty-docs"><i data-lucide="folder-open"></i> No documents indexed</li>`;
            }
            if (window.lucide) lucide.createIcons();
        } catch (e) {
            console.error("Failed to load documents", e);
        }
    }

    async function loadMCPTools() {
        try {
            const res = await fetch("/api/v1/mcp/tools");
            const data = await res.json();
            mcpToolsList.innerHTML = "";
            if (data.tools && data.tools.length > 0) {
                toolsCountBadge.textContent = `${data.tools.length} Active`;
                data.tools.forEach(tool => {
                    const li = document.createElement("li");
                    li.innerHTML = `<i data-lucide="wrench" style="width: 13px; height: 13px; color: #10a37f;"></i> <code>${escapeHTML(tool.name)}</code>`;
                    mcpToolsList.appendChild(li);
                });
            }
            if (window.lucide) lucide.createIcons();
        } catch (e) {
            console.error("Failed to load MCP tools", e);
        }
    }

    // 9. Fullscreen Drag & Drop Handling
    window.addEventListener("dragenter", (e) => {
        e.preventDefault();
        dragOverlay.classList.remove("hidden");
    });

    dragOverlay.addEventListener("dragover", (e) => e.preventDefault());

    dragOverlay.addEventListener("dragleave", (e) => {
        if (e.relatedTarget === null) {
            dragOverlay.classList.add("hidden");
        }
    });

    dragOverlay.addEventListener("drop", (e) => {
        e.preventDefault();
        dragOverlay.classList.add("hidden");
        if (e.dataTransfer.files.length > 0) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    // 10. Global Suggested Query Helper
    window.sendSuggestedQuery = function(query) {
        userPromptInput.value = query;
        userPromptInput.style.height = "auto";
        submitBtn.disabled = false;
        chatForm.dispatchEvent(new Event("submit"));
    };

    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, tag => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "'": "&#39;",
            "\"": "&quot;"
        }[tag] || tag));
    }
});
