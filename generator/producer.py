"""
Kafka Producer for Real-time ICU Patient Telemetry and BP Event Streams.
"""
import sys
import os
import time
import json
import random
import argparse
import yaml
from datetime import datetime, timezone
from typing import List, Dict, Any
from kafka import KafkaProducer

from generator.patient_state import Patient
from generator.artifacts import ArtifactInjector

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_simulation(num_patients: int = 100, speed_factor: float = 1.0, max_seconds: int = 0):
    cfg = load_config()
    kafka_servers = cfg["kafka"]["bootstrap_servers"]
    topic_telemetry = cfg["kafka"]["topics"]["telemetry"]["name"]
    topic_bp = cfg["kafka"]["topics"]["bp_events"]["name"]

    prob_scenario = cfg["generator"].get("deterioration_probability_per_tick", 0.001)
    dq_cfg = cfg["generator"].get("data_quality", {})
    injector = ArtifactInjector(
        null_rate=dq_cfg.get("null_rate", 0.01),
        spike_rate=dq_cfg.get("spike_rate", 0.005),
        late_rate=dq_cfg.get("late_record_rate", 0.02),
        late_delay_sec=dq_cfg.get("late_record_delay_seconds", 10)
    )

    ground_truth_dir = cfg["paths"]["ground_truth_dir"]
    os.makedirs(ground_truth_dir, exist_ok=True)
    gt_file_path = os.path.join(ground_truth_dir, "ground_truth_labels.csv")
    gt_file_exists = os.path.exists(gt_file_path)
    gt_file = open(gt_file_path, "a", buffering=1, encoding="utf-8")
    if not gt_file_exists:
        gt_file.write("patient_id,bed_id,event_time,risk_label,scenario,scenario_phase\n")

    print(f"Connecting KafkaProducer to {kafka_servers}...")
    producer = KafkaProducer(
        bootstrap_servers=kafka_servers,
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
        linger_ms=10,
        batch_size=32768
    )

    print(f"Initializing {num_patients} ICU ward patients...")
    patients: List[Patient] = [Patient.create(i + 1) for i in range(num_patients)]
    print(f"Streaming data at {speed_factor}x speed... (Press Ctrl+C to stop)")

    ticks = 0
    start_wall_time = time.time()

    try:
        while True:
            tick_start = time.perf_counter()
            now_iso = datetime.now(timezone.utc).isoformat()

            telemetry_batch = []
            bp_batch = []

            for p in patients:
                p.maybe_trigger_scenario(probability=prob_scenario)
                p.tick_ar1_step(phi=0.85)

                # Ground truth logging
                scenario_name = p.active_scenario.name if p.active_scenario else "none"
                scenario_phase = p.active_scenario.phase if p.active_scenario else "stable"
                gt_file.write(f"{p.patient_id},{p.bed_id},{now_iso},{p.is_deteriorating},{scenario_name},{scenario_phase}\n")

                # Telemetry record (1 Hz)
                telemetry_rec = {
                    "patient_id": p.patient_id,
                    "bed_id": p.bed_id,
                    "device_id": p.device_id,
                    "event_time": now_iso,
                    "heart_rate": p.current_vitals["heart_rate"],
                    "spo2": p.current_vitals["spo2"],
                    "respiratory_rate": p.current_vitals["respiratory_rate"],
                    "risk_label": p.is_deteriorating
                }
                rec_injected, _ = injector.process_telemetry(telemetry_rec)
                telemetry_batch.append((p.patient_id, rec_injected))

                # Intermittent BP Cuff Reading
                if p.next_bp_seconds_remaining <= 0:
                    bp_rec = {
                        "patient_id": p.patient_id,
                        "bed_id": p.bed_id,
                        "device_id": p.device_id,
                        "event_time": now_iso,
                        "systolic_bp": p.current_bp["systolic_bp"],
                        "diastolic_bp": p.current_bp["diastolic_bp"],
                        "mean_arterial_pressure": p.current_bp["mean_arterial_pressure"],
                        "risk_label": p.is_deteriorating
                    }
                    bp_batch.append((p.patient_id, bp_rec))
                    # Reset BP countdown: 1-2 min if alerted/deteriorating, 3-5 min if stable
                    if p.is_deteriorating:
                        p.next_bp_seconds_remaining = random.randint(60, 120)
                    else:
                        p.next_bp_seconds_remaining = random.randint(180, 300)

            # Transmit to Kafka
            for pid, payload in telemetry_batch:
                producer.send(topic_telemetry, key=pid, value=payload)

            for pid, payload in bp_batch:
                producer.send(topic_bp, key=pid, value=payload)

            ticks += 1
            if ticks % 10 == 0:
                producer.flush()
                deteriorating_count = sum(p.is_deteriorating for p in patients)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Tick #{ticks}: Emitted {len(telemetry_batch)} telemetry msgs, {len(bp_batch)} BP events. Active Deteriorating Patients: {deteriorating_count}/{num_patients}")

            if max_seconds > 0 and (time.time() - start_wall_time) >= max_seconds:
                print(f"Reached max duration of {max_seconds} seconds.")
                break

            # Sleep pacing to maintain 1 Hz tick (adjusted by speed_factor)
            elapsed = time.perf_counter() - tick_start
            sleep_duration = max(0.0, (1.0 / speed_factor) - elapsed)
            time.sleep(sleep_duration)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        producer.flush()
        producer.close()
        gt_file.close()
        print("Kafka Producer cleanly closed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICU Ward Data Generator")
    parser.add_argument("--patients", type=int, default=100, help="Number of concurrent patients")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier (1.0 = real-time)")
    parser.add_argument("--duration", type=int, default=0, help="Run duration in seconds (0 = infinite)")
    args = parser.parse_args()

    run_simulation(num_patients=args.patients, speed_factor=args.speed, max_seconds=args.duration)
