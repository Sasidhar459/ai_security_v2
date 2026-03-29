"""
shared_state.py

Thin module that holds globals shared between security_system.py and flask_app.py.
Fixes the __main__ vs module-name circular-import problem that caused the black
video screen — both files import from HERE instead of from each other.
"""

import numpy as np
from typing import Optional

# Updated by security_system.process_frame() — consumed by flask_app gen_frames()
latest_frame: Optional[np.ndarray] = None

# Updated by security_system.handle_intruder_alert() — consumed by flask_app /alerts
latest_alert: str = ""

# Set by security_system.py after IntruderDB is instantiated
# Consumed by flask_app.py /stop_alert and /resume_alert routes
intruder_db = None