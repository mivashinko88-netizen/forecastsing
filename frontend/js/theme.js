// theme.js - Global dark theme initialization for TrucastAI
// Sets data-theme="dark" on the document root element on page load

(function() {
    'use strict';

    // Apply dark theme immediately to prevent flash of light theme
    document.documentElement.setAttribute('data-theme', 'dark');

    // Also set on body when DOM is ready (for CSS that targets body)
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            document.body.setAttribute('data-theme', 'dark');
        });
    } else {
        // DOM already loaded
        if (document.body) {
            document.body.setAttribute('data-theme', 'dark');
        }
    }
})();
