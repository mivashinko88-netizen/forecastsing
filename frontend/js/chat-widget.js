// chat-widget.js - Floating AI Chat Widget

class ChatWidget {
    constructor() {
        this.isOpen = false;
        this.conversationHistory = [];
        this.businessId = localStorage.getItem('current_business_id');
        this.init();
    }

    init() {
        // Create widget HTML
        this.createWidget();
        // Bind events
        this.bindEvents();
    }

    createWidget() {
        // Create container
        const container = document.createElement('div');
        container.id = 'chatWidget';
        container.innerHTML = `
            <!-- Chat Button -->
            <button class="chat-button" id="chatButton" title="AI Assistant">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="chat-icon-open">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                    <circle cx="12" cy="10" r="1"/>
                    <circle cx="8" cy="10" r="1"/>
                    <circle cx="16" cy="10" r="1"/>
                </svg>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="chat-icon-close">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
            </button>

            <!-- Chat Window -->
            <div class="chat-window" id="chatWindow">
                <div class="chat-header">
                    <div class="chat-header-info">
                        <div class="chat-avatar">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 2a10 10 0 1 0 10 10H12V2z"/>
                                <circle cx="12" cy="12" r="3"/>
                            </svg>
                        </div>
                        <div>
                            <div class="chat-title">AI Assistant</div>
                            <div class="chat-status" id="chatStatus">Online</div>
                        </div>
                    </div>
                    <button class="chat-minimize" id="chatMinimize">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="5" y1="12" x2="19" y2="12"/>
                        </svg>
                    </button>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="chat-welcome">
                        <div class="chat-welcome-icon">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M12 2a10 10 0 1 0 10 10H12V2z"/>
                                <circle cx="12" cy="12" r="3"/>
                            </svg>
                        </div>
                        <h3>Hi there!</h3>
                        <p>I'm your AI assistant. Ask me anything about your forecasts, sales data, or upcoming events.</p>
                        <div class="chat-suggestions">
                            <button class="chat-suggestion" data-question="What's my busiest day next week?">Busiest day next week?</button>
                            <button class="chat-suggestion" data-question="What events are happening this month?">Events this month?</button>
                            <button class="chat-suggestion" data-question="How accurate is my model?">Model accuracy?</button>
                        </div>
                    </div>
                </div>
                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="chatInput" placeholder="Ask me anything..." autocomplete="off">
                    <button class="chat-send" id="chatSend" disabled>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="22" y1="2" x2="11" y2="13"/>
                            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                        </svg>
                    </button>
                </div>
            </div>
        `;

        // Add styles
        const styles = document.createElement('style');
        styles.textContent = `
            #chatWidget {
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 9999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }

            .chat-button {
                width: 56px;
                height: 56px;
                border-radius: 28px;
                background: linear-gradient(135deg, #14B8A6, #5EEAD4);
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(20, 184, 166, 0.4);
                transition: all 0.2s ease;
            }

            .chat-button:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 20px rgba(20, 184, 166, 0.5);
            }

            .chat-button svg {
                width: 24px;
                height: 24px;
                color: white;
            }

            .chat-button .chat-icon-close {
                display: none;
            }

            .chat-button.open .chat-icon-open {
                display: none;
            }

            .chat-button.open .chat-icon-close {
                display: block;
            }

            .chat-window {
                position: absolute;
                bottom: 70px;
                right: 0;
                width: 380px;
                height: 520px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
                display: none;
                flex-direction: column;
                overflow: hidden;
            }

            .chat-window.open {
                display: flex;
                animation: slideUp 0.2s ease;
            }

            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .chat-header {
                padding: 16px;
                background: linear-gradient(135deg, #14B8A6, #5EEAD4);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .chat-header-info {
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .chat-avatar {
                width: 40px;
                height: 40px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .chat-avatar svg {
                width: 22px;
                height: 22px;
                color: white;
            }

            .chat-title {
                font-size: 15px;
                font-weight: 600;
                color: white;
            }

            .chat-status {
                font-size: 12px;
                color: rgba(255, 255, 255, 0.8);
            }

            .chat-minimize {
                width: 32px;
                height: 32px;
                background: rgba(255, 255, 255, 0.2);
                border: none;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.15s;
            }

            .chat-minimize:hover {
                background: rgba(255, 255, 255, 0.3);
            }

            .chat-minimize svg {
                width: 18px;
                height: 18px;
                color: white;
            }

            .chat-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                background: #F9FAFB;
            }

            .chat-welcome {
                text-align: center;
                padding: 24px 16px;
            }

            .chat-welcome-icon {
                width: 56px;
                height: 56px;
                background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 16px;
            }

            .chat-welcome-icon svg {
                width: 28px;
                height: 28px;
                color: #14B8A6;
            }

            .chat-welcome h3 {
                font-size: 18px;
                font-weight: 600;
                color: #111827;
                margin-bottom: 8px;
            }

            .chat-welcome p {
                font-size: 14px;
                color: #6B7280;
                line-height: 1.5;
                margin-bottom: 16px;
            }

            .chat-suggestions {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }

            .chat-suggestion {
                padding: 10px 16px;
                background: white;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                font-size: 13px;
                color: #374151;
                cursor: pointer;
                transition: all 0.15s;
                text-align: left;
            }

            .chat-suggestion:hover {
                border-color: #14B8A6;
                background: #EEF2FF;
            }

            .chat-message {
                margin-bottom: 12px;
                display: flex;
                flex-direction: column;
            }

            .chat-message.user {
                align-items: flex-end;
            }

            .chat-message.assistant {
                align-items: flex-start;
            }

            .chat-bubble {
                max-width: 85%;
                padding: 12px 16px;
                border-radius: 16px;
                font-size: 14px;
                line-height: 1.5;
            }

            .chat-message.user .chat-bubble {
                background: linear-gradient(135deg, #14B8A6, #5EEAD4);
                color: white;
                border-bottom-right-radius: 4px;
            }

            .chat-message.assistant .chat-bubble {
                background: white;
                color: #374151;
                border-bottom-left-radius: 4px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            }

            .chat-typing {
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 12px 16px;
            }

            .chat-typing-dot {
                width: 8px;
                height: 8px;
                background: #9CA3AF;
                border-radius: 50%;
                animation: typing 1.4s ease-in-out infinite;
            }

            .chat-typing-dot:nth-child(2) {
                animation-delay: 0.2s;
            }

            .chat-typing-dot:nth-child(3) {
                animation-delay: 0.4s;
            }

            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                    opacity: 0.4;
                }
                30% {
                    transform: translateY(-4px);
                    opacity: 1;
                }
            }

            .chat-input-container {
                padding: 16px;
                background: white;
                border-top: 1px solid #E5E7EB;
                display: flex;
                gap: 8px;
            }

            .chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #E5E7EB;
                border-radius: 24px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.15s;
            }

            .chat-input:focus {
                border-color: #14B8A6;
            }

            .chat-input::placeholder {
                color: #9CA3AF;
            }

            .chat-send {
                width: 44px;
                height: 44px;
                background: #14B8A6;
                border: none;
                border-radius: 22px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.15s;
            }

            .chat-send:hover:not(:disabled) {
                background: #4F46E5;
            }

            .chat-send:disabled {
                background: #E5E7EB;
                cursor: not-allowed;
            }

            .chat-send svg {
                width: 18px;
                height: 18px;
                color: white;
            }

            .chat-send:disabled svg {
                color: #9CA3AF;
            }

            .chat-error {
                font-size: 13px;
                color: #EF4444;
                background: #FEF2F2;
                padding: 8px 12px;
                border-radius: 8px;
                margin-bottom: 12px;
            }

            @media (max-width: 480px) {
                #chatWidget {
                    bottom: 16px;
                    right: 16px;
                }

                .chat-window {
                    width: calc(100vw - 32px);
                    height: calc(100vh - 120px);
                    bottom: 64px;
                }
            }
        `;

        document.head.appendChild(styles);
        document.body.appendChild(container);
    }

    bindEvents() {
        // Toggle chat window
        document.getElementById('chatButton').addEventListener('click', () => this.toggle());
        document.getElementById('chatMinimize').addEventListener('click', () => this.toggle());

        // Send message
        const input = document.getElementById('chatInput');
        const sendBtn = document.getElementById('chatSend');

        input.addEventListener('input', () => {
            sendBtn.disabled = !input.value.trim();
        });

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && input.value.trim()) {
                this.sendMessage(input.value.trim());
            }
        });

        sendBtn.addEventListener('click', () => {
            if (input.value.trim()) {
                this.sendMessage(input.value.trim());
            }
        });

        // Suggestion buttons
        document.querySelectorAll('.chat-suggestion').forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.dataset.question;
                this.sendMessage(question);
            });
        });
    }

    toggle() {
        this.isOpen = !this.isOpen;
        document.getElementById('chatButton').classList.toggle('open', this.isOpen);
        document.getElementById('chatWindow').classList.toggle('open', this.isOpen);

        if (this.isOpen) {
            document.getElementById('chatInput').focus();
        }
    }

    async sendMessage(message) {
        const messagesContainer = document.getElementById('chatMessages');
        const input = document.getElementById('chatInput');

        // Hide welcome message
        const welcome = messagesContainer.querySelector('.chat-welcome');
        if (welcome) {
            welcome.style.display = 'none';
        }

        // Add user message
        this.addMessage('user', message);
        input.value = '';
        document.getElementById('chatSend').disabled = true;

        // Show typing indicator
        const typingId = this.showTyping();

        try {
            // Add to conversation history
            this.conversationHistory.push({
                role: 'user',
                content: message
            });

            // Call API
            const result = await api.chat(
                message,
                this.conversationHistory.slice(-10), // Last 10 messages
                this.businessId
            );

            // Remove typing indicator
            this.removeTyping(typingId);

            if (result && result.success) {
                this.addMessage('assistant', result.content);
                this.conversationHistory.push({
                    role: 'assistant',
                    content: result.content
                });
            } else if (result && !result.available) {
                this.addMessage('error', 'AI chat is currently unavailable. Please start Ollama with "ollama serve" to enable AI features.');
            } else {
                this.addMessage('error', 'Unable to get a response. Please try again.');
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.removeTyping(typingId);
            this.addMessage('error', 'Failed to send message. Please try again.');
        }
    }

    addMessage(type, content) {
        const messagesContainer = document.getElementById('chatMessages');

        const messageDiv = document.createElement('div');

        if (type === 'error') {
            messageDiv.className = 'chat-error';
            messageDiv.textContent = content;
        } else {
            messageDiv.className = `chat-message ${type}`;
            messageDiv.innerHTML = `<div class="chat-bubble">${this.escapeHtml(content)}</div>`;
        }

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showTyping() {
        const messagesContainer = document.getElementById('chatMessages');
        const typingDiv = document.createElement('div');
        const id = 'typing-' + Date.now();
        typingDiv.id = id;
        typingDiv.className = 'chat-message assistant';
        typingDiv.innerHTML = `
            <div class="chat-bubble chat-typing">
                <div class="chat-typing-dot"></div>
                <div class="chat-typing-dot"></div>
                <div class="chat-typing-dot"></div>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return id;
    }

    removeTyping(id) {
        const typingDiv = document.getElementById(id);
        if (typingDiv) {
            typingDiv.remove();
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize chat widget when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Only init if user is authenticated
        if (localStorage.getItem('access_token')) {
            window.chatWidget = new ChatWidget();
        }
    });
} else {
    // Only init if user is authenticated
    if (localStorage.getItem('access_token')) {
        window.chatWidget = new ChatWidget();
    }
}
