"""Model selection shared by Mid training and complexity measurements."""
from .TickNet import build_TickNet
from .ticknet_l import ARCHITECTURE_REVISION, build_ticknet_l


MODEL_NAMES = ("basic", "small", "large", "l")
MODEL_REVISIONS = {name: f"ticknet-{name}-upstream" for name in MODEL_NAMES[:-1]}
MODEL_REVISIONS["l"] = ARCHITECTURE_REVISION


def build_mid_model(name, *, num_classes=5, variant):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    if variant not in ("Mid32", "Mid224"):
        raise ValueError(f"Unknown dataset variant: {variant}")
    if name == "l":
        return build_ticknet_l(num_classes, cifar=variant == "Mid32")
    return build_TickNet(num_classes, typesize=name, cifar=variant == "Mid32")
