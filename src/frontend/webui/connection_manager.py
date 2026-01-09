"""
Centralized connection state manager for API connectivity.
Prevents multiple error dialogs and provides restoration notifications.
"""
import time
import gradio as gr

class ConnectionState:
    def __init__(self):
        self.is_connected = True
        self.startup_time = time.time()
        self.last_error_time = 0
        self.last_restore_time = 0
        self.last_success_time = time.time()
        self.error_shown = False
        self.consecutive_failures = 0
        self.min_uptime_before_warning = 3  # Don't show warnings for first 3 seconds after startup
        self.failure_threshold = 3  # Need 3 consecutive failures before showing warning
        self.error_cooldown = 10  # seconds between error notifications
        self.restore_cooldown = 15  # seconds between restore notifications
    
    def mark_disconnected(self, show_ui_feedback=True):
        """Mark API as disconnected and optionally show error once"""
        current_time = time.time()
        was_connected = self.is_connected
        self.is_connected = False
        self.consecutive_failures += 1
        
        # Only show error if:
        # 1. UI feedback is requested
        # 2. We've been up for at least min_uptime_before_warning (avoid startup noise)
        # 3. We have enough consecutive failures (avoid transient blips)
        # 4. Cooldown period has passed
        uptime = current_time - self.startup_time
        should_show = (
            show_ui_feedback and 
            uptime > self.min_uptime_before_warning and
            self.consecutive_failures >= self.failure_threshold and
            (current_time - self.last_error_time > self.error_cooldown)
        )
        
        if should_show and not self.error_shown:
            gr.Warning("⚠️ Connection to API server lost. Will retry automatically...")
            self.error_shown = True
            self.last_error_time = current_time
        
        return False
    
    def mark_connected(self, show_ui_feedback=True):
        """Mark API as connected and optionally show restoration notice"""
        current_time = time.time()
        was_disconnected = not self.is_connected
        self.is_connected = True
        self.last_success_time = current_time
        self.consecutive_failures = 0
        
        # Show green success message only if:
        # 1. UI feedback is requested
        # 2. We were previously disconnected AND had shown an error
        # 3. We haven't shown a restore message recently (cooldown)
        if show_ui_feedback and was_disconnected and self.error_shown:
            if current_time - self.last_restore_time > self.restore_cooldown:
                gr.Info("✅ Connection restored!")
                self.last_restore_time = current_time
                self.error_shown = False
        
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
