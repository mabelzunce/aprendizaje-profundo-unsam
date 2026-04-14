"""
=============================================================================
 YOLO v1 - You Only Look Once: Unified, Real-Time Object Detection
 *** STUDENT VERSION — Fill in the TODO sections ***
=============================================================================

 Paper: Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016).
        "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.

 This script has the same structure as the full demo, but several key
 sections are left as TODOs for you to implement.

 Sections to complete:
   TODO 1: Target encoding (encode bounding boxes into the grid)
   TODO 2: YOLO v1 model architecture
   TODO 3: Loss function (multi-part YOLO loss)
   TODO 4: IoU computation
   TODO 5: Non-Maximum Suppression (NMS)
   TODO 6: Decoding predictions from the output tensor

 Author: Aprendizaje Profundo - UNSAM
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import random
import time

# ── Reproducibility ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ===========================================================================
# YOLO HYPER-PARAMETERS
# ===========================================================================
CLASSES = ["circle", "rectangle", "triangle"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 128
S = 7          # Grid size
B = 2          # Bounding boxes per cell
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5


# ===========================================================================
# PROVIDED — Synthetic Data Generation (no changes needed)
# ===========================================================================
def random_color():
    return tuple(random.randint(100, 255) for _ in range(3))


def generate_sample(img_size=IMG_SIZE, max_objects=3):
    """
    Generate one synthetic image with random shapes and their bounding boxes.
    Returns:
        image : PIL.Image (RGB, img_size × img_size)
        boxes : list of [class_id, x_center, y_center, width, height]  (normalised 0-1)
    """
    img = Image.new("RGB", (img_size, img_size), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    boxes = []

    n_objects = random.randint(1, max_objects)

    for _ in range(n_objects):
        cls = random.randint(0, NUM_CLASSES - 1)
        color = random_color()

        obj_w = random.randint(15, 45)
        obj_h = random.randint(15, 45)
        x1 = random.randint(0, img_size - obj_w)
        y1 = random.randint(0, img_size - obj_h)
        x2 = x1 + obj_w
        y2 = y1 + obj_h

        if cls == 0:
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif cls == 1:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        elif cls == 2:
            cx = (x1 + x2) / 2
            draw.polygon([(cx, y1), (x1, y2), (x2, y2)], fill=color)

        xc = ((x1 + x2) / 2) / img_size
        yc = ((y1 + y2) / 2) / img_size
        w = obj_w / img_size
        h = obj_h / img_size
        boxes.append([cls, xc, yc, w, h])

    return img, boxes


# ===========================================================================
# TODO 1 — DATASET WITH TARGET ENCODING
# ===========================================================================
class SyntheticDetectionDataset(Dataset):
    """Generates a synthetic dataset on the fly."""

    def __init__(self, num_samples=1000, img_size=IMG_SIZE, S=S, B=B,
                 num_classes=NUM_CLASSES, max_objects=3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.S = S
        self.B = B
        self.C = num_classes
        self.max_objects = max_objects
        self.data = [generate_sample(img_size, max_objects)
                     for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, boxes = self.data[idx]

        # Image → tensor [3, H, W] normalised to [0, 1]
        img_tensor = torch.tensor(
            np.array(img), dtype=torch.float32
        ).permute(2, 0, 1) / 255.0

        # Encode boxes into YOLO target tensor
        target = self.encode_target(boxes)

        return img_tensor, target

    def encode_target(self, boxes):
        """
        ┌─────────────────────────────────────────────────────────────┐
        │  TODO 1: Encode bounding boxes into a YOLO target tensor   │
        └─────────────────────────────────────────────────────────────┘

        Convert a list of [class_id, xc, yc, w, h] (all normalised 0-1)
        into a tensor of shape (S, S, 5*B + C).

        Steps:
          1. Create a zeros tensor of shape (S, S, 5*B + C).
          2. For each bounding box:
             a. Determine which grid cell (grid_x, grid_y) contains
                the box centre.
             b. Compute x_offset and y_offset *within that cell* (0-1).
             c. For each of the B bounding box slots, store:
                [x_offset, y_offset, w, h, 1.0 (confidence)]
             d. Set the one-hot class vector in the last C entries.
          3. Only assign the first object to each cell (if two objects
             fall in the same cell, keep the first).

        Hints:
          - grid_x = int(xc * S),  clamped to [0, S-1]
          - x_offset = xc * S - grid_x
          - Check target[grid_y, grid_x, 4] == 0 before assigning
            (to avoid overwriting).
        """
        target = torch.zeros(self.S, self.S, 5 * self.B + self.C)

        for box in boxes:
            cls_id, xc, yc, w, h = box

            # ── YOUR CODE HERE ──────────────────────────────────────
            # 1. Compute grid_x and grid_y

            # 2. Compute x_offset and y_offset within the cell

            # 3. Check if the cell is free (target[grid_y, grid_x, 4] == 0)

            # 4. Fill B box slots with [x_offset, y_offset, w, h, 1.0]

            # 5. Set the one-hot class: target[grid_y, grid_x, 5*B + cls_id] = 1.0

            pass
            # ── END YOUR CODE ───────────────────────────────────────

        return target


# ===========================================================================
# TODO 2 — YOLO v1 MODEL
# ===========================================================================
class YOLOv1(nn.Module):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  TODO 2: Build the YOLO v1 model architecture                  │
    └─────────────────────────────────────────────────────────────────┘

    Build a CNN that takes images of shape (batch, 3, 128, 128) and
    outputs a tensor of shape (batch, S, S, 5*B + C).

    Recommended architecture:
      Backbone (feature extractor):
        5 blocks of: Conv2d(3×3, padding=1) → BatchNorm2d → LeakyReLU(0.1) → MaxPool2d(2,2)
        Channels: 3 → 32 → 64 → 128 → 256 → 512
        After 5 pools of stride 2: 128 / 2^5 = 4, so output is (512, 4, 4)

      Head (detection layers):
        Flatten → Linear(512*4*4, 1024) → LeakyReLU(0.1) → Dropout(0.5)
        → Linear(1024, S*S*(5*B+C))

    Don't forget to reshape the output to (batch, S, S, 5*B + C) at the end!

    Hints:
      - nn.Sequential makes building the backbone easy.
      - Use view(-1, S, S, 5*B+C) to reshape.
    """

    def __init__(self, S=7, B=2, C=3):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        # ── YOUR CODE HERE ──────────────────────────────────────────
        # 1. Define self.backbone: 5 conv blocks with pooling

        # 2. Define self.head: Flatten → Linear → activation → Dropout → Linear

        pass
        # ── END YOUR CODE ───────────────────────────────────────────

    def forward(self, x):
        # ── YOUR CODE HERE ──────────────────────────────────────────
        # 1. Pass through backbone
        # 2. Pass through head
        # 3. Reshape to (batch, S, S, 5*B + C)
        # 4. Return
        pass
        # ── END YOUR CODE ───────────────────────────────────────────


# ===========================================================================
# TODO 3 — YOLO v1 LOSS FUNCTION
# ===========================================================================
class YOLOv1Loss(nn.Module):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  TODO 3: Implement the YOLO v1 loss function                   │
    └─────────────────────────────────────────────────────────────────┘

    The YOLO loss has multiple components:

      L = λ_coord · Σ_obj [ (x-x̂)² + (y-ŷ)² ]               — centre loss
        + λ_coord · Σ_obj [ (√w-√ŵ)² + (√h-√ĥ)² ]           — size loss (sqrt!)
        + Σ_obj   [ (C - Ĉ)² ]                               — confidence (object)
        + λ_noobj · Σ_noobj [ (C - Ĉ)² ]                     — confidence (no object)
        + Σ_obj   [ Σ_c (p_c - p̂_c)² ]                       — class probabilities

    Input shapes: predictions and targets are both (batch, S, S, 5*B + C)

    Steps:
      1. Build an obj_mask where target confidence (index 4) > 0.5
      2. Build noobj_mask = ~obj_mask
      3. Compute coordinate loss (xy + wh with sqrt) for cells with objects
      4. Compute confidence loss for object/no-object cells
      5. Compute classification loss for cells with objects
      6. Sum all components, divide by batch_size

    Hints:
      - Use obj_mask.unsqueeze(-1) to broadcast over the last dimension.
      - For the sqrt trick: torch.sqrt(torch.abs(pred_wh) + 1e-8)
      - predictions[..., 0:2] gives x, y of box 0
      - predictions[..., 4] gives confidence of box 0
      - predictions[..., 9] gives confidence of box 1
      - predictions[..., 5*B:] gives class probabilities
    """

    def __init__(self, S=7, B=2, C=3, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]

        # ── YOUR CODE HERE ──────────────────────────────────────────
        # 1. Create obj_mask and noobj_mask

        # 2. Coordinate loss (xy)

        # 3. Coordinate loss (wh with sqrt trick)

        # 4. Confidence loss for object cells (both box 0 and box 1)

        # 5. Confidence loss for no-object cells (scaled by lambda_noobj)

        # 6. Classification loss

        # 7. Total loss = sum of all parts / batch_size

        total_loss = torch.tensor(0.0, requires_grad=True)  # placeholder
        return total_loss
        # ── END YOUR CODE ───────────────────────────────────────────


# ===========================================================================
# TODO 4 — IoU COMPUTATION
# ===========================================================================
def compute_iou(box1, box2):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  TODO 4: Compute Intersection over Union (IoU)                 │
    └─────────────────────────────────────────────────────────────────┘

    Each box is [x_center, y_center, width, height] (normalised 0-1).

    Steps:
      1. Convert both boxes from (xc, yc, w, h) to corner format
         (x1, y1, x2, y2).
      2. Compute the intersection rectangle:
         - inter_x1 = max(b1_x1, b2_x1)
         - inter_y1 = max(b1_y1, b2_y1)
         - inter_x2 = min(b1_x2, b2_x2)
         - inter_y2 = min(b1_y2, b2_y2)
      3. Intersection area = max(0, inter_x2 - inter_x1) * max(0, ...)
      4. Union area = area1 + area2 - intersection
      5. IoU = intersection / (union + epsilon)

    Returns: float (IoU value)
    """

    # ── YOUR CODE HERE ──────────────────────────────────────────────

    return 0.0  # placeholder
    # ── END YOUR CODE ───────────────────────────────────────────────


# ===========================================================================
# TODO 5 — NON-MAXIMUM SUPPRESSION
# ===========================================================================
def non_max_suppression(detections, iou_threshold=0.4):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  TODO 5: Implement Non-Maximum Suppression (NMS)               │
    └─────────────────────────────────────────────────────────────────┘

    detections: list of [class_id, score, xc, yc, w, h]

    Algorithm:
      1. Sort detections by score (descending).
      2. Pick the detection with the highest score → add to "kept" list.
      3. Remove all remaining detections of the SAME CLASS that have
         IoU > iou_threshold with the picked detection.
      4. Repeat until no detections remain.

    Returns: filtered list of detections.

    Hints:
      - det[0] is the class_id
      - det[2:6] is [xc, yc, w, h] — pass to compute_iou()
    """

    if len(detections) == 0:
        return []

    # ── YOUR CODE HERE ──────────────────────────────────────────────

    return detections  # placeholder
    # ── END YOUR CODE ───────────────────────────────────────────────


# ===========================================================================
# TODO 6 — DECODE PREDICTIONS
# ===========================================================================
def decode_predictions(output, S=7, B=2, C=3, conf_threshold=0.3):
    """
    ┌─────────────────────────────────────────────────────────────────┐
    │  TODO 6: Decode the YOLO output tensor into bounding boxes     │
    └─────────────────────────────────────────────────────────────────┘

    output: tensor of shape (S, S, 5*B + C)

    For each grid cell (gx, gy):
      1. Get the class probabilities from output[gy, gx, 5*B:]
      2. Find the best class and its probability.
      3. For each of the B bounding boxes:
         a. Read x_offset, y_offset, w, h, confidence from the box slot.
         b. Compute final score = class_prob * confidence.
         c. If score > conf_threshold:
            - Convert x_offset back to absolute normalised coords:
              xc = (gx + x_offset) / S
              yc = (gy + y_offset) / S
            - Append [class_id, score, xc, yc, |w|, |h|] to detections.

    Returns: list of [class_id, score, xc, yc, w, h]
    """
    detections = []

    # ── YOUR CODE HERE ──────────────────────────────────────────────

    # ── END YOUR CODE ───────────────────────────────────────────────

    return detections


# ===========================================================================
# PROVIDED — Training Loop (no changes needed)
# ===========================================================================
def train_yolo(model, train_loader, val_loader, device, epochs=40, lr=1e-3):
    criterion = YOLOv1Loss(S=S, B=B, C=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {"train_loss": [], "val_loss": []}

    print(f"\n{'='*60}")
    print(f" Training YOLO v1  —  {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs}  |  "
                  f"Train: {train_loss:.4f}  |  Val: {val_loss:.4f}  |  "
                  f"LR: {lr_now:.6f}")

    print(f"{'='*60}\n")
    return history


# ===========================================================================
# PROVIDED — Visualisation functions (no changes needed)
# ===========================================================================
def visualise_samples(dataset, n=6):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    cmap = ["cyan", "lime", "orange"]

    for i in range(n):
        img_t, target = dataset[i]
        img_np = img_t.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(f"Sample {i}", fontsize=12)
        axes[i].axis("off")

        for gy in range(S):
            for gx in range(S):
                if target[gy, gx, 4] > 0.5:
                    x_off = target[gy, gx, 0].item()
                    y_off = target[gy, gx, 1].item()
                    w = target[gy, gx, 2].item()
                    h = target[gy, gx, 3].item()
                    xc = (gx + x_off) / S * IMG_SIZE
                    yc = (gy + y_off) / S * IMG_SIZE
                    bw, bh = w * IMG_SIZE, h * IMG_SIZE
                    cls_id = target[gy, gx, 5 * B:].argmax().item()
                    rect = patches.Rectangle(
                        (xc - bw / 2, yc - bh / 2), bw, bh,
                        linewidth=2, edgecolor=cmap[cls_id], facecolor="none"
                    )
                    axes[i].add_patch(rect)
                    axes[i].text(
                        xc - bw / 2, yc - bh / 2 - 3,
                        CLASSES[cls_id], color=cmap[cls_id],
                        fontsize=9, weight="bold",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1)
                    )

    plt.suptitle("Synthetic Dataset — Ground Truth", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def visualise_detections(model, dataset, device, n=6,
                          conf_threshold=0.25, iou_threshold=0.4):
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    cmap = ["cyan", "lime", "orange"]

    for i in range(n):
        img_t, _ = dataset[i]
        img_np = img_t.permute(1, 2, 0).numpy()
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(device))[0].cpu()

        detections = decode_predictions(pred, S=S, B=B, C=NUM_CLASSES,
                                         conf_threshold=conf_threshold)
        detections = non_max_suppression(detections, iou_threshold=iou_threshold)

        axes[i].imshow(img_np)
        axes[i].set_title(f"Sample {i} — {len(detections)} det(s)", fontsize=11)
        axes[i].axis("off")

        for det in detections:
            cls_id, score, xc, yc, w, h = det
            color = cmap[cls_id]
            px, py = xc * IMG_SIZE, yc * IMG_SIZE
            pw, ph = w * IMG_SIZE, h * IMG_SIZE
            rect = patches.Rectangle(
                (px - pw / 2, py - ph / 2), pw, ph,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            axes[i].add_patch(rect)
            axes[i].text(
                px - pw / 2, py - ph / 2 - 4,
                f"{CLASSES[cls_id]} {score:.2f}",
                color=color, fontsize=9, weight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1)
            )

    plt.suptitle("YOLO v1 — Detections after NMS", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# MAIN — Run everything
# ===========================================================================
if __name__ == "__main__":
    print("Generating synthetic dataset...")
    train_dataset = SyntheticDetectionDataset(num_samples=2000, max_objects=3)
    val_dataset = SyntheticDetectionDataset(num_samples=300, max_objects=3)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Target shape: ({S}, {S}, {5*B + NUM_CLASSES})")

    # Show some samples (works if encode_target is implemented)
    visualise_samples(train_dataset, n=6)

    # Build model
    model = YOLOv1(S=S, B=B, C=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE).to(device)
    out = model(dummy)
    print(f"Forward test: {dummy.shape} → {out.shape}")

    # Train
    history = train_yolo(model, train_loader, val_loader, device,
                          epochs=40, lr=1e-3)

    # Plot training
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("YOLO v1 Training Curves")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # Visualise detections
    visualise_detections(model, val_dataset, device, n=6)

    print("\n✅ All TODOs implemented successfully!" if out is not None
          else "\n⚠️  Check your implementations.")
