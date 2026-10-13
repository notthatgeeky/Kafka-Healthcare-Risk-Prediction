"""
Data quality challenges and sensor artifact injector.
"""
import random
from typing import Dict, Any, Tuple

class ArtifactInjector:
    def __init__(self, null_rate: float = 0.01, spike_rate: float = 0.005, late_rate: float = 0.02, late_delay_sec: int = 10):
        self.null_rate = null_rate
        self.spike_rate = spike_rate
        self.late_rate = late_rate
        self.late_delay_sec = late_delay_sec

    def process_telemetry(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Injects motion spikes or nulls. Returns (record, delay_seconds).
        """
        delay = 0

        # 1. Late arrival simulation
        if random.random() < self.late_rate:
            delay = random.randint(3, self.late_delay_sec)

        # 2. Motion artifact spike
        if random.random() < self.spike_rate:
            target = random.choice(["heart_rate", "spo2", "respiratory_rate"])
            if target == "heart_rate":
                record["heart_rate"] = float(random.choice([210.0, 25.0]))
            elif target == "spo2":
                record["spo2"] = float(random.choice([45.0, 100.0]))
            elif target == "respiratory_rate":
                record["respiratory_rate"] = float(random.choice([55.0, 2.0]))

        # 3. Missing sensor values / lead off
        if random.random() < self.null_rate:
            target = random.choice(["heart_rate", "spo2", "respiratory_rate"])
            record[target] = None

        return record, delay
