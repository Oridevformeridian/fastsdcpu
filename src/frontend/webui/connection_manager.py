"""
Centralized connection state manager for API connectivity.
Prevents multiple error dialogs and provides restoration notifications.
"""
import time
import gradio as gr

class ConnectionState:
    def __init__(self):
        self.is_connected = True
        self.last_error_time = 0
        self.error_shown = False
        self.error_cooldown = 5  # seconds between error notifications
        self.reload_on_restore = True  # Auto-reload page when connection restored
    
    def mark_disconnected(self, show_ui_feedback=True):
        """Mark API as disconnected and optionally show error once"""
        current_time = time.time()
        was_connected = self.is_connected
        self.is_connected = False
        
        # Only show error if:
        # 1. UI feedback is requested
        # 2. We haven't shown an error recently (cooldown)
        # 3. We transitioned from connected to disconnected (not already shown)
        if show_ui_feedback and (current_time - self.last_error_time > self.error_cooldown):
            if was_connected or not self.error_shown:
                gr.Warning("⚠️ Connection to API server lost. Will retry automatically...")
                self.error_shown = True
                self.last_error_time = current_time
        
        return False
    
    def mark_connected(self, show_ui_feedback=True):
        """Mark API as connected and optionally show restoration notice with page reload"""
        was_disconnected = not self.is_connected
        self.is_connected = True
        
        # Show green success message if we were previously disconnected
        if show_ui_feedback and was_disconnected and self.error_shown:
            gr.Info("✅ Connection restored! Reloading page to get latest updates...")
            self.error_shown = False
            
            # Trigger page reload if enabled
            if self.reload_on_restore:
                # Return special marker that UI can use to trigger reload
                return "RELOAD_PAGE"
        
        return True
    
    def check_and_update(self, api_succeeded, show_ui_feedback=True):
        """Update state based on API call result and return current state"""
        if api_succeeded:
            return self.mark_connected(show_ui_feedback)
        else:
            return self.mark_disconnected(show_ui_feedback)

# Global singleton instance
_connection_state = ConnectionState()

def get_connection_state():
    """Get the global connection state instance"""
    return _connection_state
