/**
 * Dashboard notification fix for the AI Trading Agent Dashboard
 * 
 * This script fixes the notification system by:
 * 1. Safely handling DOM operations
 * 2. Preventing errors when elements don't exist
 * 3. Providing a safe notification API
 */

// Execute immediately on load
(function() {
    console.log('Initializing safe notification system...');
    
    // Safe function to add a notification
    window.addNotification = function(notification) {
        try {
            if (!notification) {
                console.warn('Invalid notification object');
                return false;
            }
            
            const notificationList = document.getElementById('notificationList');
            if (!notificationList) {
                console.warn('Notification list element not found');
                return false;
            }
            
            // Create notification item
            const notificationItem = document.createElement('div');
            
            // Apply classes safely
            notificationItem.className = 'notification-item';
            if (notification.read === false) {
                try {
                    if (notificationItem.classList) {
                        notificationItem.classList.add('unread');
                    }
                } catch (e) {
                    console.warn('Error adding unread class:', e);
                }
            }
            
            // Set notification ID
            if (notification.id) {
                try {
                    notificationItem.setAttribute('data-id', notification.id);
                } catch (e) {
                    console.warn('Error setting notification ID:', e);
                }
            }
            
            // Set notification content
            try {
                notificationItem.innerHTML = `
                    <div class="notification-header">
                        <div class="notification-title">${notification.title || 'Notification'}</div>
                        <div class="notification-time">${notification.time || new Date().toLocaleTimeString()}</div>
                    </div>
                    <div class="notification-content">${notification.message || ''}</div>
                `;
            } catch (e) {
                console.warn('Error setting notification content:', e);
                notificationItem.textContent = notification.message || 'Notification';
            }
            
            // Add to list
            try {
                notificationList.appendChild(notificationItem);
            } catch (e) {
                console.warn('Error appending notification to list:', e);
                return false;
            }
            
            return true;
        } catch (error) {
            console.error('Error in addNotification:', error);
            return false;
        }
    };
    
    // Safe function to show a notification
    window.showNotification = function(message, type, duration) {
        try {
            if (!message) return false;
            
            type = type || 'info';
            duration = duration || 5000;
            
            // Look for toast container or create one
            let toastContainer = document.getElementById('toastContainer');
            if (!toastContainer) {
                toastContainer = document.createElement('div');
                toastContainer.id = 'toastContainer';
                
                // Apply styles
                toastContainer.style.position = 'fixed';
                toastContainer.style.top = '20px';
                toastContainer.style.right = '20px';
                toastContainer.style.zIndex = '9999';
                
                // Append to body
                document.body.appendChild(toastContainer);
            }
            
            // Create toast
            const toast = document.createElement('div');
            
            // Apply styles
            toast.style.backgroundColor = type === 'error' ? '#f44336' : 
                                        type === 'warning' ? '#ff9800' : 
                                        type === 'success' ? '#4caf50' : '#2196f3';
            toast.style.color = '#fff';
            toast.style.padding = '16px';
            toast.style.borderRadius = '4px';
            toast.style.marginBottom = '10px';
            toast.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
            toast.style.minWidth = '300px';
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s ease-in-out';
            
            // Set content
            toast.textContent = message;
            
            // Add to container
            toastContainer.appendChild(toast);
            
            // Show with animation
            setTimeout(() => {
                toast.style.opacity = '1';
            }, 10);
            
            // Auto remove
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.style.opacity = '0';
                    setTimeout(() => {
                        if (toast.parentNode) {
                            toastContainer.removeChild(toast);
                        }
                    }, 300);
                }
            }, duration);
            
            return true;
        } catch (error) {
            console.error('Error in showNotification:', error);
            return false;
        }
    };
    
    console.log('Safe notification system initialized successfully!');
})();
