from __future__ import annotations
import datetime as _dt

def run_id_now() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
