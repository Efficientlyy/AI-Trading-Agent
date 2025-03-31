/**
 * Add Bitvavo Menu Item
 * 
 * This script adds a Bitvavo settings menu item to the dashboard navigation.
 * Add this script to the bottom of the modern_dashboard.html template.
 */

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function () {
    // Find the settings dropdown menu
    const settingsDropdown = document.querySelector('.dropdown-menu');

    if (!settingsDropdown) {
        console.error('Settings dropdown menu not found');
        return;
    }

    // Check if Bitvavo settings are already in the menu
    if (document.querySelector('[data-settings-type="bitvavo"]')) {
        console.log('Bitvavo settings already in menu');
        return;
    }

    // Create the Bitvavo settings menu item
    const bitvavoMenuItem = document.createElement('li');
    bitvavoMenuItem.innerHTML = `
        <a class="dropdown-item" href="#" data-bs-toggle="modal" data-bs-target="#settingsModal" 
           data-settings-type="bitvavo" onclick="loadBitvavoSettings()">
            <i class="fas fa-exchange-alt me-2"></i>Bitvavo Settings
        </a>
    `;

    // Add the menu item to the dropdown
    settingsDropdown.appendChild(bitvavoMenuItem);

    console.log('Added Bitvavo settings to menu');

    // Add the loadBitvavoSettings function if it doesn't exist
    if (typeof loadBitvavoSettings === 'undefined') {
        window.loadBitvavoSettings = function () {
            fetch('/api/templates/bitvavo_settings_panel.html')
                .then(response => response.text())
                .then(html => {
                    document.getElementById('settingsModalBody').innerHTML = html;
                    document.getElementById('settingsModalLabel').innerText = 'Bitvavo Settings';

                    // Load the Bitvavo settings CSS
                    if (!document.getElementById('bitvavo-settings-css')) {
                        const link = document.createElement('link');
                        link.id = 'bitvavo-settings-css';
                        link.rel = 'stylesheet';
                        link.href = '/static/css/bitvavo_settings.css';
                        document.head.appendChild(link);
                    }
                })
                .catch(error => {
                    console.error('Error loading Bitvavo settings:', error);
                    document.getElementById('settingsModalBody').innerHTML =
                        '<div class="alert alert-danger">Error loading Bitvavo settings</div>';
                });
        };

        console.log('Added loadBitvavoSettings function');
    }
});