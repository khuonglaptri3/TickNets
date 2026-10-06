# Interactive Architecture Diagrams: TickNets

This directory contains interactive architectural diagrams detailing tensor flow, Pointwise-Depthwise-Pointwise (PDP) blocks, channel bottlenecking, Mixed Depthwise Convolutions, and parameter / FLOP distributions across all 3 model families in the project.

All diagrams are self-contained HTML files that can be opened directly in any modern web browser (Chrome, Firefox, Safari, Edge) without local servers or external web frameworks.

---

## Architecture Diagram Inventory

| Architecture | Interactive Diagram (.html) | Input Configuration (.json) | Visualized Architecture Highlights |
| :--- | :--- | :--- | :--- |
| **TickNet-Basic** | [ticknet_basic.html](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_basic.html) | `ticknet_basic.architecture.json` | 1.06M Params, 0.988 GFLOPs. Visualizes the Stage 2 computational bottleneck (52.7% of total FLOPs). |
| **TickNet-L v1** | [ticknet_l.html](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_l.html) | `ticknet_l.architecture.json` | 1.10M Params, 0.796 GFLOPs (-19.4%). Highlights **Pointwise Bottleneck (0.75x)** and **Mixed DW (3x3 + 5x5)** multi-scale receptive field expansion. |
| **TickNet-C v1** | [ticknet_c.html](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_c.html) | `ticknet_c.architecture.json` | 5.16M Params, 0.821 GFLOPs. Inherits original author backbone, expanding to 9 blocks with Stage 5 reaching 896 channels under stride schedule `(2, 1, 2, 2, 2)`. |
| **Macro Comparison** | [ticknet_comparison.html](file:///home/intern-tdkhuong/Desktop/TickNets/docs/architectures/ticknet_comparison.html) | `ticknet_comparison.architecture.json` | Holistic comparison of 3 design philosophies: Baseline vs. Compute-Efficient vs. Capacity-Maximized. |

---

## Instructions for Opening in Web Browsers

Open any HTML diagram file using standard methods:

### Option 1: Open Directly via Terminal (Ubuntu / Linux)

```bash
# Open TickNet-Basic diagram
xdg-open docs/architectures/ticknet_basic.html

# Open TickNet-L v1 diagram
xdg-open docs/architectures/ticknet_l.html

# Open TickNet-C v1 diagram
xdg-open docs/architectures/ticknet_c.html

# Open Macro Comparison diagram
xdg-open docs/architectures/ticknet_comparison.html
```

### Option 2: Drag and Drop into Browser

Drag the `.html` file from your file manager (Nautilus/Files) directly into any browser window (Chrome, Edge, Firefox).

---

## Interactive Features in Each Diagram

1. **Pan & Zoom**: Scroll with mouse wheel or trackpad gestures to inspect individual layers within blocks or zoom out for end-to-end global flow.
2. **Light & Dark Theme Switcher**: Dedicated theme toggle button located in the corner, optimized for slide decks or print-ready reports.
3. **Metadata Badges**: Displays intermediate tensor shapes `(B, C, H, W)` per stage, kernel sizes, strides, parameter counts, and respective FLOPs.
4. **Showcase Quality Compliance**: All diagrams adhere to strict presentation criteria (9/9 Archify validation standards passed, label clearances $\ge 80$px, zero viewport overflow).
