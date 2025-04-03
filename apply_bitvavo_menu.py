"""
Apply Bitvavo Menu

This script adds the Bitvavo tab to the settings modal.
"""

import os
import sys
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_bitvavo_tab():
    """Add the Bitvavo tab to the settings modal."""
    try:
        # Path to the settings_modal.html template
        template_path = os.path.join('templates', 'settings_modal.html')
        
        # Check if the template exists
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            return False
        
        # Read the template file
        with open(template_path, 'r') as f:
            template_content = f.read()
        
        # Check if Bitvavo settings are already in the modal
        if 'data-tab="bitvavo"' in template_content:
            logger.info("Bitvavo settings already in settings modal")
            return True
        
        # Find the settings tabs
        tabs_pattern = r'<div class="settings-tabs">(.*?)</div>'
        tabs_match = re.search(tabs_pattern, template_content, re.DOTALL)
        
        if not tabs_match:
            logger.error("Settings tabs not found in template")
            return False
        
        # Add the Bitvavo tab
        tabs_content = tabs_match.group(1)
        bitvavo_tab = '\n                    <button class="settings-tab" data-tab="bitvavo">Bitvavo</button>'
        new_tabs_content = tabs_content + bitvavo_tab
        
        # Replace the tabs content
        new_template_content = template_content.replace(tabs_content, new_tabs_content)
        
        # Find the settings content
        content_pattern = r'<div class="settings-content">(.*?)</div>\s*</div>\s*</div>\s*<div class="modal-footer">'
        content_match = re.search(content_pattern, new_template_content, re.DOTALL)
        
        if not content_match:
            logger.error("Settings content not found in template")
            return False
        
        # Add the Bitvavo settings pane
        content = content_match.group(1)
        bitvavo_pane = '''
                    <!-- Bitvavo Settings Tab -->
                    <div class="settings-pane" id="bitvavo-settings">
                        <div id="bitvavo-settings-container">
                            <!-- Bitvavo settings will be loaded dynamically -->
                            <div class="loading-placeholder">Loading Bitvavo settings...</div>
                        </div>
                    </div>'''
        
        new_content = content + bitvavo_pane
        
        # Replace the content
        final_template_content = new_template_content.replace(content, new_content)
        
        # Write the updated template back to the file
        with open(template_path, 'w') as f:
            f.write(final_template_content)
        
        logger.info("Added Bitvavo tab to settings modal")
        
        # Add the loadBitvavoSettings function to the settings_panel.js file
        js_path = os.path.join('static', 'js', 'settings_panel.js')
        
        if not os.path.exists(js_path):
            logger.error(f"JavaScript file not found: {js_path}")
            return False
        
        # Read the JavaScript file
        with open(js_path, 'r') as f:
            js_content = f.read()
        
        # Check if the loadBitvavoSettings function is already in the file
        if 'function loadBitvavoSettings()' in js_content:
            logger.info("loadBitvavoSettings function already in settings_panel.js")
        else:
            # Find the class definition
            class_pattern = r'class SettingsPanelController {'
            class_match = re.search(class_pattern, js_content)
            
            if not class_match:
                logger.error("SettingsPanelController class not found in settings_panel.js")
                return False
            
            # Find the constructor
            constructor_pattern = r'constructor\(\) {(.*?)}'
            constructor_match = re.search(constructor_pattern, js_content, re.DOTALL)
            
            if not constructor_match:
                logger.error("Constructor not found in settings_panel.js")
                return False
            
            # Add the loadBitvavoSettings method to the class
            method_pattern = r'showReloadNotification\(\) {.*?}\s*}'
            method_match = re.search(method_pattern, js_content, re.DOTALL)
            
            if not method_match:
                logger.error("showReloadNotification method not found in settings_panel.js")
                return False
            
            # Add the loadBitvavoSettings method
            bitvavo_method = '''

    /**
     * Load Bitvavo settings panel
     */
    loadBitvavoSettings() {
        fetch('/api/templates/bitvavo_settings_panel.html')
            .then(response => response.text())
            .then(html => {
                const container = document.getElementById('bitvavo-settings-container');
                if (container) {
                    container.innerHTML = html;
                    
                    // Load the Bitvavo settings CSS
                    if (!document.getElementById('bitvavo-settings-css')) {
                        const link = document.createElement('link');
                        link.id = 'bitvavo-settings-css';
                        link.rel = 'stylesheet';
                        link.href = '/static/css/bitvavo_settings.css';
                        document.head.appendChild(link);
                    }
                    
                    // Initialize the Bitvavo settings component
                    if (typeof BitvavoSettingsComponent !== 'undefined') {
                        const bitvavoSettings = new BitvavoSettingsComponent('bitvavo-settings-container');
                        bitvavoSettings.initialize();
                    }
                }
            })
            .catch(error => {
                console.error('Error loading Bitvavo settings:', error);
                const container = document.getElementById('bitvavo-settings-container');
                if (container) {
                    container.innerHTML = '<div class="alert alert-danger">Error loading Bitvavo settings</div>';
                }
            });
    }'''
            
            # Insert the method before the end of the class
            method_end = method_match.end()
            new_js_content = js_content[:method_end] + bitvavo_method + js_content[method_end:]
            
            # Write the updated JavaScript back to the file
            with open(js_path, 'w') as f:
                f.write(new_js_content)
            
            logger.info("Added loadBitvavoSettings method to settings_panel.js")
        
        # Add event listener for the Bitvavo tab
        if 'case \'bitvavo\':' not in js_content:
            # Find the tab click handler
            tab_pattern = r'// Handle tab clicks'
            tab_match = re.search(tab_pattern, js_content)
            
            if not tab_match:
                logger.error("Tab click handler not found in settings_panel.js")
                return False
            
            # Find the switch statement
            switch_pattern = r'switch\(tabId\) {(.*?)}'
            switch_match = re.search(switch_pattern, js_content, re.DOTALL)
            
            if not switch_match:
                logger.error("Switch statement not found in settings_panel.js")
                return False
            
            # Add the Bitvavo case
            switch_content = switch_match.group(1)
            bitvavo_case = '''
                case 'bitvavo':
                    this.loadBitvavoSettings();
                    break;'''
            
            new_switch_content = switch_content + bitvavo_case
            
            # Replace the switch content
            final_js_content = new_js_content.replace(switch_content, new_switch_content)
            
            # Write the updated JavaScript back to the file
            with open(js_path, 'w') as f:
                f.write(final_js_content)
            
            logger.info("Added Bitvavo tab event listener to settings_panel.js")
        
        return True
    except Exception as e:
        logger.error(f"Error adding Bitvavo tab: {e}")
        return False

if __name__ == "__main__":
    logger.info("Adding Bitvavo tab to settings modal...")
    
    if add_bitvavo_tab():
        logger.info("Successfully added Bitvavo tab")
    else:
        logger.error("Failed to add Bitvavo tab")
        sys.exit(1)