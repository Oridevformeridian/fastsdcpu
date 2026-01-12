from threading import Lock

# Global lock to serialize pipeline mutations and generation.
pipeline_lock = Lock()
