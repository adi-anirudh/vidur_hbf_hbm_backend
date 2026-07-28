#!/usr/bin/env python3
"""Write the exact paper evaluation contract to a versionable JSON artifact."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation_common import BATCHES, PLATFORM, TP_CHOICES, TR_SENSITIVITY_NS


def main() -> None:
    out = Path("results/evaluation_setup.json")
    payload = PLATFORM.manifest()
    payload["batch_choices"] = list(BATCHES)
    payload["tp_choices"] = list(TP_CHOICES)
    payload["read_latency_sensitivity_ns"] = list(TR_SENSITIVITY_NS)
    payload["read_latency_sensitivity_effective_bw_gbps"] = {
        str(int(tr)): PLATFORM.effective_hbf_bw_gbps(tr)
        for tr in TR_SENSITIVITY_NS
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
