"""
Patient state representation with physiological baselines, AR(1) dynamics, and deterioration modeling.
"""
import random
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from generator.scenarios import ScenarioState, create_scenario

# Physiological clamping boundaries
CLAMP_RANGES = {
    "heart_rate": (30.0, 220.0),
    "spo2": (50.0, 100.0),
    "respiratory_rate": (4.0, 50.0),
    "systolic_bp": (50.0, 240.0),
    "diastolic_bp": (30.0, 150.0),
}

@dataclass
class Patient:
    patient_id: str
    bed_id: str
    device_id: str
    baselines: Dict[str, float]
    current_vitals: Dict[str, float]
    current_bp: Dict[str, float]
    admitted: bool = True
    active_scenario: Optional[ScenarioState] = None
    next_bp_seconds_remaining: int = 120
    cooldown_seconds: int = 0

    @classmethod
    def create(cls, patient_num: int) -> 'Patient':
        pid = f"P{patient_num:04d}"
        bid = f"BED-{patient_num:03d}"
        dev_id = f"MON-{patient_num:03d}"

        # Clinical baseline sampling
        base_hr = float(np.clip(np.random.normal(75.0, 8.0), 55.0, 95.0))
        base_spo2 = float(np.clip(np.random.beta(50, 2) * 100, 94.0, 99.5))
        base_rr = float(np.clip(np.random.normal(16.0, 2.0), 12.0, 20.0))
        base_sbp = float(np.clip(np.random.normal(120.0, 10.0), 100.0, 145.0))
        base_dbp = float(np.clip(np.random.normal(75.0, 7.0), 60.0, 90.0))

        baselines = {
            "heart_rate": round(base_hr, 1),
            "spo2": round(base_spo2, 1),
            "respiratory_rate": round(base_rr, 1),
            "systolic_bp": round(base_sbp, 1),
            "diastolic_bp": round(base_dbp, 1),
        }

        current_vitals = {
            "heart_rate": baselines["heart_rate"],
            "spo2": baselines["spo2"],
            "respiratory_rate": baselines["respiratory_rate"],
        }

        current_bp = {
            "systolic_bp": baselines["systolic_bp"],
            "diastolic_bp": baselines["diastolic_bp"],
            "mean_arterial_pressure": round((2 * baselines["diastolic_bp"] + baselines["systolic_bp"]) / 3.0, 1),
        }

        return cls(
            patient_id=pid,
            bed_id=bid,
            device_id=dev_id,
            baselines=baselines,
            current_vitals=current_vitals,
            current_bp=current_bp,
            next_bp_seconds_remaining=random.randint(60, 240)
        )

    def tick_ar1_step(self, phi: float = 0.85):
        """
        Executes one 1-second simulation step with AR(1) mean-reverting stochastic process.
        """
        # 1. Update active scenario
        scenario_shifts = {}
        is_deteriorating = False
        if self.active_scenario:
            is_deteriorating = True
            self.active_scenario.elapsed_seconds += 1
            for vital in ["heart_rate", "spo2", "respiratory_rate", "systolic_bp", "diastolic_bp"]:
                scenario_shifts[vital] = self.active_scenario.get_current_shift(vital)

            if self.active_scenario.is_finished:
                self.active_scenario = None
                self.cooldown_seconds = random.randint(60, 180)  # cooldown before next event
        elif self.cooldown_seconds > 0:
            self.cooldown_seconds -= 1

        # 2. Coupling effects (e.g. low SpO2 drives higher RR and HR)
        hypoxia_comp = max(0.0, (93.0 - self.current_vitals["spo2"]) * 0.7) if self.current_vitals["spo2"] < 93.0 else 0.0

        noise_stds = {
            "heart_rate": 0.8,
            "spo2": 0.25,
            "respiratory_rate": 0.35,
            "systolic_bp": 1.2,
            "diastolic_bp": 0.8,
        }

        # 3. Update telemetry vitals
        for vital in ["heart_rate", "spo2", "respiratory_rate"]:
            target = self.baselines[vital] + scenario_shifts.get(vital, 0.0)
            if vital == "respiratory_rate":
                target += hypoxia_comp
            elif vital == "heart_rate":
                target += (hypoxia_comp * 1.5)

            noise = np.random.normal(0, noise_stds[vital])
            prev = self.current_vitals[vital]
            new_val = target + phi * (prev - target) + noise
            c_min, c_max = CLAMP_RANGES[vital]
            self.current_vitals[vital] = round(float(np.clip(new_val, c_min, c_max)), 1)

        # 4. Update BP vitals
        for bp_key in ["systolic_bp", "diastolic_bp"]:
            target = self.baselines[bp_key] + scenario_shifts.get(bp_key, 0.0)
            noise = np.random.normal(0, noise_stds[bp_key])
            prev = self.current_bp[bp_key]
            new_val = target + phi * (prev - target) + noise
            c_min, c_max = CLAMP_RANGES[bp_key]
            self.current_bp[bp_key] = round(float(np.clip(new_val, c_min, c_max)), 1)

        sbp = self.current_bp["systolic_bp"]
        dbp = self.current_bp["diastolic_bp"]
        self.current_bp["mean_arterial_pressure"] = round((2.0 * dbp + sbp) / 3.0, 1)

        # 5. Decrement BP countdown
        self.next_bp_seconds_remaining -= 1

    def maybe_trigger_scenario(self, probability: float = 0.001):
        if self.active_scenario is None and self.cooldown_seconds == 0:
            if random.random() < probability:
                choice = random.choice(["hypoxia", "tachycardia", "hypotension", "sepsis_like"])
                self.active_scenario = create_scenario(choice)

    @property
    def is_deteriorating(self) -> int:
        return 1 if self.active_scenario is not None else 0
