
import json
import subprocess
import re
import time
from datetime import datetime
from pathlib import Path

OUTPUT = Path("results/live_telemetry.jsonl")
INTERVAL = 30
PING_COUNT = 5
TARGET = "8.8.8.8"


def get_gateway():
    result = subprocess.run(
        ["route", "-n", "get", "default"],
        capture_output=True, text=True, timeout=5
    )
    match = re.search(r"gateway:\s*(\S+)", result.stdout)
    return match.group(1) if match else None


def ping_host(host):
    try:
        result = subprocess.run(
            ["ping", "-c", str(PING_COUNT), host],
            capture_output=True, text=True, timeout=20
        )
        output = result.stdout

        loss_match = re.search(
            r"([\d.]+)% packet loss", output
        )
        rtt_match = re.search(
            r"round-trip min/avg/max/stddev = "
            r"([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", output
        )

        return {
            "reachable": loss_match is not None
                         and float(loss_match.group(1)) < 100,
            "packet_loss_pct": (
                float(loss_match.group(1))
                if loss_match else None
            ),
            "min_ms": float(rtt_match.group(1))
                      if rtt_match else None,
            "avg_ms": float(rtt_match.group(2))
                      if rtt_match else None,
            "max_ms": float(rtt_match.group(3))
                      if rtt_match else None,
        }
    except (subprocess.TimeoutExpired, OSError):
        return {
            "reachable": False,
            "packet_loss_pct": None,
            "min_ms": None,
            "avg_ms": None,
            "max_ms": None,
        }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    gateway = get_gateway()

    print("Live network collector started")
    print("Gateway:", gateway)
    print("Target:", TARGET)
    print("Stop with Ctrl+C\n")

    try:
        while True:
            timestamp = datetime.now().astimezone().isoformat()
            record = {
                "timestamp": timestamp,
                "gateway": gateway,
                "gateway_ping": (
                    ping_host(gateway) if gateway else None
                ),
                "internet_ping": ping_host(TARGET),
            }

            with OUTPUT.open("a") as f:
                f.write(json.dumps(record) + "\n")

            print(json.dumps(record, indent=2))
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\nCollector stopped.")


if __name__ == "__main__":
    main()
