/**
 * User Menu Functionality
 * 
 * This module provides functionality for the user menu dropdown in the dashboard header.
 * It handles user profile, settings, and logout actions.
 */

document.addEventListener('DOMContentLoaded', function () {
    console.log('User menu dropdown behavior fixed by adding proper CSS transitions and click/hover handling');

    // Initialize user menu
    initializeUserMenu();
});

function initializeUserMenu() {
    // Get DOM elements
    const userMenuButton = document.getElementById('user-menu-button');
    const userMenuDropdown = document.getElementById('user-menu-dropdown');

    // Check if elements exist
    if (!userMenuButton || !userMenuDropdown) {
        console.warn('User menu elements not found');
        return;
    }

    // Toggle dropdown on button click
    userMenuButton.addEventListener('click', function (event) {
        event.stopPropagation();
        userMenuDropdown.classList.toggle('show');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', function (event) {
        if (!userMenuButton.contains(event.target) && !userMenuDropdown.contains(event.target)) {
            userMenuDropdown.classList.remove('show');
        }
    });

    // Handle menu item clicks
    setupMenuItemHandlers();
}

function setupMenuItemHandlers() {
    // Profile link
    const profileLink = document.getElementById('user-profile-link');
    if (profileLink) {
        profileLink.addEventListener('click', function (event) {
            event.preventDefault();
            showUserProfile();
        });
    }

    // Settings link
    const settingsLink = document.getElementById('user-settings-link');
    if (settingsLink) {
        settingsLink.addEventListener('click', function (event) {
            event.preventDefault();
            showUserSettings();
        });
    }

    // Logout link
    const logoutLink = document.getElementById('user-logout-link');
    if (logoutLink) {
        logoutLink.addEventListener('click', function (event) {
            event.preventDefault();
            logoutUser();
        });
    }
}

function showUserProfile() {
    console.log('Showing user profile');
    // In a real implementation, this would show a user profile modal or navigate to a profile page
}

function showUserSettings() {
    console.log('Showing user settings');
    // In a real implementation, this would show a settings modal or navigate to a settings page
}

function logoutUser() {
    console.log('Logging out user');
    // In a real implementation, this would send a logout request to the server
    // and then redirect to the login page
    window.location.href = '/logout';
}
