"""The motion models the responsibility metrics can use, behind one set of
command-line options:

  densetnt  CAT's pretrained DenseTNT (responsibility/densetnt.py; needs
            TensorFlow for its input parsing)
  mtr       UniTraj's MTR trained on WOMD (responsibility/unitraj.py; needs
            UniTraj and its CUDA ops, i.e. a GPU), with --checkpoint

Models are imported only when chosen, so neither brings in the other's
dependencies.
"""

from pathlib import Path
from typing import Dict, Optional

MODELS = ("densetnt", "mtr")


def add_model_arguments(parser) -> None:
    g = parser.add_argument_group("motion model")
    g.add_argument("--model", default="densetnt", choices=MODELS)
    g.add_argument("--checkpoint", default=None, help="mtr: the trained MTR (UniTraj / Lightning .ckpt).")
    g.add_argument("--unitraj-root", default=None, help="mtr: the UniTraj checkout (default: $UNITRAJ_ROOT).")
    g.add_argument("--unitraj-method", default=None,
                   help="mtr: the method config it was trained with (default: MTR_womd).")
    g.add_argument("--temperature", type=float, default=None,
                   help="mtr: softmax temperature of the intention scores (default: the checkpoint's "
                        "calibration file, else 1).")


def model_settings(args) -> Dict:
    """What identifies the model's results (stored in a run's config.json)."""
    if args.model == "densetnt":
        return {"name": "densetnt"}
    return {"name": "mtr", "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "method": args.unitraj_method or "MTR_womd", "temperature": resolved_temperature(args)}


def resolved_temperature(args) -> Optional[float]:
    if args.model != "mtr":
        return None
    if args.temperature is not None:
        return float(args.temperature)
    from responsibility.unitraj import load_calibration

    return load_calibration(args.checkpoint) if args.checkpoint else 1.0


def load_model(args, device: str):
    if args.model == "densetnt":
        from responsibility.densetnt import DenseTNT

        return DenseTNT(device=device)
    if not args.checkpoint:
        raise SystemExit("--model mtr needs --checkpoint")
    from responsibility.unitraj import DEFAULT_METHOD, Inputs, UniTrajModel, load_config, mtr_predictor

    config = load_config(method=args.unitraj_method or DEFAULT_METHOD, root=args.unitraj_root)
    predictor = mtr_predictor(args.checkpoint, config, device=device, root=args.unitraj_root)
    return UniTrajModel(predictor, Inputs(config, root=args.unitraj_root), temperature=resolved_temperature(args))
