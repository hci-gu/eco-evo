"""Shared spatial boundary option for training, probes and inference."""


def add_boundary_argument(parser):
    parser.add_argument("--boundary", choices=("bounded", "torus"), default="bounded",
                        help="Torus wraps movement and observations, replacing migration; "
                             "bounded keeps --migration semantics (default)")


def validate_boundary(boundary):
    if boundary not in ("bounded", "torus"):
        raise ValueError("Boundary must be bounded or torus")
    return boundary
