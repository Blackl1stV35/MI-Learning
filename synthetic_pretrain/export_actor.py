#!/usr/bin/env python3
"""
Export DSAC actor weights.

Usage:
    # Validate + export JSON for Rust ndarray loader (required):
    python export_actor.py --checkpoint models/dsac_pretrained.pt --json models/actor_weights.json

    # Optional ONNX export for Python bridge fallback:
    python export_actor.py --checkpoint models/dsac_pretrained.pt --onnx models/actor.onnx
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent))


EXPECTED_KEYS = {
    "actor.net.0.weight": (512, 118),
    "actor.net.0.bias":   (512,),
    "actor.net.2.weight": (256, 512),
    "actor.net.2.bias":   (256,),
    "actor.net.4.weight": (2,   256),
    "actor.net.4.bias":   (2,),
}


def validate(ckpt_path: str) -> bool:
    ckpt     = torch.load(ckpt_path, map_location="cpu")
    actor_sd = ckpt.get("actor", {})
    flat     = {f"actor.{k}": v for k, v in actor_sd.items()}

    all_ok = True
    print(f"Validating {ckpt_path}:")
    for key, expected in EXPECTED_KEYS.items():
        if key not in flat:
            print(f"  MISSING: {key}")
            all_ok = False
            continue
        actual = tuple(flat[key].shape)
        if actual != expected:
            print(f"  SHAPE MISMATCH: {key}  expected={expected}  got={actual}")
            all_ok = False
        else:
            print(f"  OK: {key:40s} {actual}")

    if all_ok:
        print(f"\n✓ Checkpoint valid")
    else:
        print(f"\n✗ Validation failed")
    return all_ok


def export_json(ckpt_path: str, json_path: str) -> None:
    """Export actor weights to JSON for pure-Rust ndarray loader."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["actor"]
    out  = {
        "w0": sd["net.0.weight"].flatten().tolist(),
        "b0": sd["net.0.bias"].tolist(),
        "w2": sd["net.2.weight"].flatten().tolist(),
        "b2": sd["net.2.bias"].tolist(),
        "w4": sd["net.4.weight"].flatten().tolist(),
        "b4": sd["net.4.bias"].tolist(),
    }
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f)
    size_kb = Path(json_path).stat().st_size // 1024
    print(f"JSON weights exported: {json_path}  ({size_kb} KB)")
    print(f"Pass to signal server: --actor {json_path}")


def export_onnx(ckpt_path: str, onnx_path: str) -> None:
    """Optional ONNX export for Python bridge fallback."""
    sys.path.insert(0, str(Path(__file__).parent))
    from train_dsac import Actor

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    actor = Actor()
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    dummy = torch.zeros(1, 118)
    torch.onnx.export(
        actor, dummy, onnx_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX exported: {onnx_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="models/dsac_pretrained.pt")
    p.add_argument("--json",       default=None,
                   help="Export weights to JSON for Rust ndarray loader")
    p.add_argument("--onnx",       default=None,
                   help="Export to ONNX for Python bridge fallback")
    args = p.parse_args()

    ok = validate(args.checkpoint)
    if not ok:
        sys.exit(1)

    if args.json:
        export_json(args.checkpoint, args.json)

    if args.onnx:
        export_onnx(args.checkpoint, args.onnx)

    if not args.json and not args.onnx:
        print("\nNo export format specified. Add --json models/actor_weights.json")


if __name__ == "__main__":
    main()