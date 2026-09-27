"""Reproduce the Basic/L/C inference-complexity comparison without training."""
import argparse
import json
from pathlib import Path

import torch

from models.mid_models import MODEL_NAMES, MODEL_REVISIONS, build_mid_model
from models.model_profile import COUNTING_CONVENTION, EXCLUDED_OPERATIONS, profile_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=("basic", "l", "c"))
    parser.add_argument("--sizes", nargs="+", type=int, choices=(32, 224), default=(32, 224))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    measurements = []
    for name in args.models:
        for size in args.sizes:
            variant = f"Mid{size}"
            model = build_mid_model(name, variant=variant)
            measurement = profile_model(model, size, cross_check=True)
            measurement.update(model=name, variant=variant, architecture_revision=MODEL_REVISIONS[name])
            measurements.append(measurement)
            print(f"{name:5s} {variant:6s}: {measurement['learnable_parameters']:,} parameters, "
                  f"{measurement['macs'] / 1e9:.6f} GMACs, {measurement['flops'] / 1e9:.6f} GFLOPs, "
                  f"limits_pass={measurement['within_exam_limits']}")
    report = {"schema_version": 1, "torch_version": str(torch.__version__),
              "device": "cpu", "dtype": "float32", "num_classes": 5,
              "counting_convention": COUNTING_CONVENTION,
              "excluded_operations": list(EXCLUDED_OPERATIONS),
              "limits": {"learnable_parameters_max_inclusive": 6_000_000,
                         "flops_max_exclusive": 1_000_000_000},
              "l_design_target_flops": 800_000_000,
              "c_design_target_flops": 850_000_000,
              "measurements": measurements}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
