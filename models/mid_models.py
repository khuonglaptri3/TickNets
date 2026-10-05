"""Model selection shared by Mid training and complexity measurements."""
from .TickNet import build_TickNet
from .ticknet_l import ARCHITECTURE_REVISION as L_REVISION, build_ticknet_l


BASELINE_MODEL_NAMES = ("basic", "small", "large")
CUSTOM_MODEL_NAMES = ("l",)
MODEL_NAMES = BASELINE_MODEL_NAMES + CUSTOM_MODEL_NAMES
MODEL_REVISIONS = {name: f"ticknet-{name}-upstream" for name in BASELINE_MODEL_NAMES}
MODEL_REVISIONS.update(l=L_REVISION)


def build_mid_model(name, *, num_classes=5, variant):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    if variant not in ("Mid32", "Mid224"):
        raise ValueError(f"Unknown dataset variant: {variant}")
    if name == "l":
        return build_ticknet_l(num_classes, cifar=variant == "Mid32")
    return build_TickNet(num_classes, typesize=name, cifar=variant == "Mid32")
