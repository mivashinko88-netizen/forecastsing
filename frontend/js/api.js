// api.js - API Client Wrapper

const API_BASE = window.location.origin + '/api';

class ApiClient {
    constructor() {
        this.accessToken = localStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token');
    }

    // Store tokens
    setTokens(accessToken, refreshToken) {
        this.accessToken = accessToken;
        this.refreshToken = refreshToken;
        localStorage.setItem('access_token', accessToken);
        localStorage.setItem('refresh_token', refreshToken);
    }

    // Clear tokens
    clearTokens() {
        this.accessToken = null;
        this.refreshToken = null;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('user');
    }

    // Get stored user
    getUser() {
        const user = localStorage.getItem('user');
        return user ? JSON.parse(user) : null;
    }

    // Store user
    setUser(user) {
        localStorage.setItem('user', JSON.stringify(user));
    }

    // Check if authenticated
    isAuthenticated() {
        return !!this.accessToken;
    }

    // Make authenticated request
    async request(endpoint, options = {}) {
        const url = endpoint.startsWith('http') ? endpoint : `${API_BASE}${endpoint}`;

        const headers = {
            ...options.headers
        };

        // Add auth header if we have a token
        if (this.accessToken) {
            headers['Authorization'] = `Bearer ${this.accessToken}`;
        }

        // Add content type for JSON bodies
        if (options.body && !(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(options.body);
        }

        try {
            const response = await fetch(url, {
                ...options,
                headers
            });

            // Handle 401 - try to refresh token
            if (response.status === 401 && this.refreshToken) {
                const refreshed = await this.refreshAccessToken();
                if (refreshed) {
                    // Retry the request with new token
                    headers['Authorization'] = `Bearer ${this.accessToken}`;
                    const retryResponse = await fetch(url, { ...options, headers });
                    return this.handleResponse(retryResponse);
                } else {
                    // Refresh failed, redirect to login
                    this.clearTokens();
                    window.location.href = '/frontend/pages/login.html';
                    return null;
                }
            }

            return this.handleResponse(response);
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Handle response
    async handleResponse(response) {
        const contentType = response.headers.get('content-type');

        if (!response.ok) {
            let error;
            if (contentType && contentType.includes('application/json')) {
                const data = await response.json();
                error = new Error(data.detail || data.message || 'Request failed');
                error.data = data;
            } else {
                error = new Error('Request failed');
            }
            error.status = response.status;
            throw error;
        }

        if (contentType && contentType.includes('application/json')) {
            return response.json();
        }

        return response.text();
    }

    // Refresh access token
    async refreshAccessToken() {
        try {
            const response = await fetch(`${API_BASE}/auth/refresh`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ refresh_token: this.refreshToken })
            });

            if (response.ok) {
                const data = await response.json();
                this.setTokens(data.access_token, data.refresh_token);
                return true;
            }
            return false;
        } catch {
            return false;
        }
    }

    // ============ Auth Endpoints ============

    async googleAuth(credential) {
        const data = await this.request('/auth/google', {
            method: 'POST',
            body: { token: credential }
        });

        if (data) {
            this.setTokens(data.access_token, data.refresh_token);
            this.setUser(data.user);
        }

        return data;
    }

    async getMe() {
        return this.request('/auth/me');
    }

    async logout() {
        try {
            await this.request('/auth/logout', { method: 'POST' });
        } finally {
            this.clearTokens();
        }
    }

    // ============ Business Endpoints ============

    async getBusinesses() {
        return this.request('/businesses');
    }

    async createBusiness(data) {
        return this.request('/businesses', {
            method: 'POST',
            body: data
        });
    }

    async getBusiness(id) {
        return this.request(`/businesses/${id}`);
    }

    async updateBusiness(id, data) {
        return this.request(`/businesses/${id}`, {
            method: 'PUT',
            body: data
        });
    }

    async deleteBusiness(id) {
        return this.request(`/businesses/${id}`, {
            method: 'DELETE'
        });
    }

    async completeBusinessSetup(id) {
        return this.request(`/businesses/${id}/complete-setup`, {
            method: 'POST'
        });
    }

    // ============ Upload & Training Endpoints ============

    async uploadFile(businessId, file) {
        const formData = new FormData();
        formData.append('file', file);

        return this.request(`/businesses/${businessId}/upload`, {
            method: 'POST',
            body: formData
        });
    }

    async trainModel(businessId, file, columnMapping = null) {
        const formData = new FormData();
        formData.append('file', file);
        if (columnMapping) {
            formData.append('column_mapping', JSON.stringify(columnMapping));
        }

        return this.request(`/businesses/${businessId}/train`, {
            method: 'POST',
            body: formData
        });
    }

    // ============ Model Endpoints ============

    async getModels(businessId) {
        return this.request(`/businesses/${businessId}/models`);
    }

    async getModel(modelId) {
        return this.request(`/models/${modelId}`);
    }

    async activateModel(modelId) {
        return this.request(`/models/${modelId}/activate`, {
            method: 'PUT'
        });
    }

    async deleteModel(modelId) {
        return this.request(`/models/${modelId}`, {
            method: 'DELETE'
        });
    }

    // ============ Prediction Endpoints ============

    async predict(modelId, days = 7, items = null) {
        return this.request(`/models/${modelId}/predict`, {
            method: 'POST',
            body: { days, items }
        });
    }

    async getForecasts(businessId) {
        return this.request(`/businesses/${businessId}/forecasts`);
    }

    async recordActual(predictionId, actualQuantity) {
        return this.request(`/predictions/${predictionId}/actual`, {
            method: 'POST',
            body: { actual_quantity: actualQuantity }
        });
    }

    // ============ Events Endpoints ============

    async getEvents(businessId, date = null, startDate = null, endDate = null) {
        let url = `/businesses/${businessId}/events`;
        const params = new URLSearchParams();

        if (date) {
            params.append('date', date);
        } else if (startDate && endDate) {
            params.append('start_date', startDate);
            params.append('end_date', endDate);
        }

        if (params.toString()) {
            url += '?' + params.toString();
        }

        return this.request(url);
    }

    // ============ LLM Endpoints ============

    async getLLMStatus() {
        return this.request('/llm/status');
    }

    async summarizeDay(date, events, weather = null, predictions = null) {
        return this.request('/llm/summarize/day', {
            method: 'POST',
            body: { date, events, weather, predictions }
        });
    }

    async summarizeForecast(predictions, factors = null, dateRange = null) {
        return this.request('/llm/summarize/forecast', {
            method: 'POST',
            body: { predictions, factors, date_range: dateRange }
        });
    }

    async summarizeDashboard(stats, recentPredictions = null) {
        return this.request('/llm/summarize/dashboard', {
            method: 'POST',
            body: { stats, recent_predictions: recentPredictions }
        });
    }

    async chat(message, conversationHistory = null, businessId = null) {
        return this.request('/llm/chat', {
            method: 'POST',
            body: {
                message,
                conversation_history: conversationHistory,
                business_id: businessId
            }
        });
    }

    // ============ Integration Endpoints ============

    async getIntegrations(businessId) {
        return this.request(`/integrations/businesses/${businessId}`);
    }

    async getIntegration(integrationId) {
        return this.request(`/integrations/${integrationId}`);
    }

    async disconnectIntegration(integrationId) {
        return this.request(`/integrations/${integrationId}`, {
            method: 'DELETE'
        });
    }

    async testIntegration(integrationId) {
        return this.request(`/integrations/${integrationId}/test`, {
            method: 'POST'
        });
    }

    async triggerSync(integrationId, syncType = 'full', startDate = null, endDate = null) {
        const params = new URLSearchParams({ sync_type: syncType });
        if (startDate) params.append('start_date', startDate);
        if (endDate) params.append('end_date', endDate);

        return this.request(`/integrations/${integrationId}/sync?${params}`, {
            method: 'POST'
        });
    }

    async getSyncHistory(integrationId, limit = 10) {
        return this.request(`/integrations/${integrationId}/sync-history?limit=${limit}`);
    }

    async getSyncedDataSummary(businessId, integrationId = null) {
        let url = `/integrations/businesses/${businessId}/synced-data/summary`;
        if (integrationId) {
            url += `?integration_id=${integrationId}`;
        }
        return this.request(url);
    }

    async exportSyncedToCsv(businessId, integrationId = null) {
        let url = `/integrations/businesses/${businessId}/export-csv`;
        if (integrationId) {
            url += `?integration_id=${integrationId}`;
        }
        return this.request(url, { method: 'POST' });
    }

    async getIntegrationProviders() {
        return this.request('/integrations/providers');
    }
}

// Create singleton instance
const api = new ApiClient();

// ============ Theme Manager ============

class ThemeManager {
    constructor() {
        this.storageKey = 'theme';
        this.init();
    }

    init() {
        // Load saved theme or default to light
        const savedTheme = localStorage.getItem(this.storageKey) || 'light';
        this.setTheme(savedTheme, false);
    }

    getTheme() {
        return document.documentElement.getAttribute('data-theme') || 'light';
    }

    setTheme(theme, save = true) {
        document.documentElement.setAttribute('data-theme', theme);
        if (save) {
            localStorage.setItem(this.storageKey, theme);
        }
        // Dispatch event for any listeners
        window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
    }

    toggle() {
        const current = this.getTheme();
        const newTheme = current === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
        return newTheme;
    }

    isDark() {
        return this.getTheme() === 'dark';
    }
}

// Create theme manager instance
const themeManager = new ThemeManager();

// Export for use in other modules
window.api = api;
window.themeManager = themeManager;
