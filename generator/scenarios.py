"""
Scenarios module defining physiological deterioration trajectories.
"""
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ScenarioState:
    name: str
    phase: str  # 'ramp_up', 'peak', 'recovery'
    elapsed_seconds: int
    duration_ramp_up: int
    duration_peak: int
    duration_recovery: int
    peak_shifts: Dict[str, float]

    @property
    def total_duration(self) -> int:
        return self.duration_ramp_up + self.duration_peak + self.duration_recovery

    @property
    def is_finished(self) -> bool:
        return self.elapsed_seconds >= self.total_duration

    def get_current_shift(self, vital: str) -> float:
        """Calculates current vital shift based on progression phase."""
        peak = self.peak_shifts.get(vital, 0.0)
        if peak == 0.0:
            return 0.0

        if self.elapsed_seconds < self.duration_ramp_up:
            self.phase = "ramp_up"
            progress = self.elapsed_seconds / max(1, self.duration_ramp_up)
            return peak * progress
        elif self.elapsed_seconds < (self.duration_ramp_up + self.duration_peak):
            self.phase = "peak"
            return peak
        else:
            self.phase = "recovery"
            recovery_elapsed = self.elapsed_seconds - (self.duration_ramp_up + self.duration_peak)
            progress = 1.0 - (recovery_elapsed / max(1, self.duration_recovery))
            return peak * max(0.0, progress)


def create_scenario(name: str) -> ScenarioState:
    """Factory to create realistic ICU deterioration scenarios."""
    if name == "hypoxia":
        # Acute hypoxemic respiratory failure
        return ScenarioState(
            name="hypoxia",
            phase="ramp_up",
            elapsed_seconds=0,
            duration_ramp_up=25,
            duration_peak=60,
            duration_recovery=30,
            peak_shifts={
                "spo2": -12.0,            # e.g., 97 -> 85%
                "heart_rate": 25.0,        # compensatory tachycardia
                "respiratory_rate": 9.0,   # compensatory tachypnea
                "systolic_bp": 10.0,
                "diastolic_bp": 5.0
            }
        )
    elif name == "tachycardia":
        # Paroxysmal supraventricular / sinus tachycardia
        return ScenarioState(
            name="tachycardia",
            phase="ramp_up",
            elapsed_seconds=0,
            duration_ramp_up=20,
            duration_peak=50,
            duration_recovery=25,
            peak_shifts={
                "heart_rate": 45.0,        # e.g., 75 -> 120+ bpm
                "spo2": -2.0,
                "respiratory_rate": 4.0,
                "systolic_bp": 25.0,
                "diastolic_bp": 15.0
            }
        )
    elif name == "hypotension":
        # Acute hypovolemic or vasodilatory drop
        return ScenarioState(
            name="hypotension",
            phase="ramp_up",
            elapsed_seconds=0,
            duration_ramp_up=30,
            duration_peak=60,
            duration_recovery=30,
            peak_shifts={
                "systolic_bp": -35.0,      # e.g., 120 -> 85 mmHg (critical shock risk)
                "diastolic_bp": -20.0,
                "heart_rate": 22.0,        # reflex tachycardia
                "respiratory_rate": 4.0,
                "spo2": -3.0
            }
        )
    elif name == "sepsis_like":
        # Septic shock pattern: severe vasodilation + tachycardia + hypoxia
        return ScenarioState(
            name="sepsis_like",
            phase="ramp_up",
            elapsed_seconds=0,
            duration_ramp_up=35,
            duration_peak=90,
            duration_recovery=40,
            peak_shifts={
                "heart_rate": 35.0,        # e.g. 75 -> 110 bpm
                "systolic_bp": -30.0,      # MAP drops
                "diastolic_bp": -18.0,
                "respiratory_rate": 8.0,   # tachypneic
                "spo2": -6.0               # desaturation
            }
        )
    else:
        raise ValueError(f"Unknown scenario: {name}")
