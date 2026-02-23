# Berana Project Journal: The Journey of Columnar Layout Analysis

## 1. The Genesis of the Problem
The mission started with a deceptively simple goal: taking a 700-page ancient Ethiopic (Ge'ez and Amharic) liturgical PDF and turning it into a structured, digital database. However, the moment we opened the first file, we hit a wall.

The document wasn't just text; it was a complex 3-column "Triple" layout.
- **Column 1:** Ge'ez (Ancient script)
- **Column 2:** Amharic (Modern translation)
- **Column 3:** English (Global interpretation)

### Why Math Failed Us
Our first instinct was to use traditional mathematical "Gap-Seeking" or "Canyon Detection" (using OpenCV). We tried to find vertical white strips where the horizontal pixel projection was zero.
**It was a disaster.**
- **Physical Skew:** Ancient scans are never 90 degrees. Even a 1-degree tilt over the height of a 3000px image means the "white strip" moves 50 pixels horizontally, crashing directly into the text.
- **Ink Bleed & Noise:** Stains on the parchment created "fake text" in the gutters, while faint ink made real text look like white space.
Mathematical logic is too rigid; it couldn't "see" the intent behind the page. We realized we needed **Computer Vision**—specifically, something that could identify a "gutter" as a semantic concept.

---

## 2. The Shift to Vision (and the Setup Struggles)
We decided to train a custom model to identify the column dividers. We looked at several options:
- **Surya Layout:** Great for modern docs, but it kept merging our columns into one big mess. It was too "black-box."
- **YOLOv8 Detection:** It can draw boxes, but boxes are rectangles. Rectangles cannot follow a tilted, diagonal line.
- **YOLOv8 Segmentation (The Winner):** We chose the **Small Segmenter (YOLOv8s-seg)**. Why? Because it doesn't just draw a box; it draws a "pixel mask." It can follow a wavy or crooked line perfectly.

### The Infrastructure Headache
Before we could train, we had to build the factory. Setting up **Label Studio** was our first major environment challenge. We had to:
1. Dockerize Label Studio to keep the environment isolated.
2. Build a custom `setup.sh` that injected host UIDs into the container so file permissions wouldn't break.
3. Create a custom XML UI config just to get 1px wide lines so we could actually see what we were labeling.
4. Bridge the gap between local OS paths and Docker’s internal volume mapping.

---

## 3. The Labeling Treadmill & "Active Learning"
We started by manually labeling 30 pages. It was slow. We exported them in **YOLO 1.1 (Polygon)** format—a specific text-based coordinate system where every point is a percentage of the image width/height. We chose this over COCO JSON because YOLO's format is resilient to image resizing.

**The First Fail:** Our initial model targets were too broad. It was guessing dividers but missing the faint ones.
**The Fix:** We implemented **Active Learning**. We ran the mediocre model on 100 fresh pages, imported the "guesses" back into Label Studio, and had a human (you) rapidly fix them. This "AI-assisted labeling" is what allowed us to scale from 30 pages to 456 pages without losing our minds.

---

## 4. The v13 Validation Breakthrough
After 13 rounds of training, we hit the "Golden Run." You shared the results, and they were stunning:
- **Mask mAP50: 0.946**
- **Precision: 0.862**
- **Recall: 0.880**


#code block

```      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
     99/100      10.3G      2.548      1.089      1.283      0.918          0          2       1024: 100% ━━━━━━━━━━━━ 4/4 1.6s/it 6.3s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 7.1it/s 0.1s
                   all         12         24      0.791      0.731      0.841      0.339       0.63       0.75      0.806      0.469

      Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss   sem_loss  Instances       Size
    100/100      10.3G      1.975      1.965      1.308      1.151          0          2       1024: 100% ━━━━━━━━━━━━ 4/4 2.2s/it 8.8s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 6.8it/s 0.1s
                   all         12         24      0.703       0.75      0.833       0.34      0.805      0.705      0.794      0.466

100 epochs completed in 0.270 hours.
Optimizer stripped from /home/ephrem/projects/berana/runs/segment/runs/layout/berana_divider_v13/weights/last.pt, 23.9MB
Optimizer stripped from /home/ephrem/projects/berana/runs/segment/runs/layout/berana_divider_v13/weights/best.pt, 23.9MB

Validating /home/ephrem/projects/berana/runs/segment/runs/layout/berana_divider_v13/weights/best.pt...
Ultralytics 8.4.14 🚀 Python-3.12.3 torch-2.10.0+cu130 CUDA:0 (NVIDIA GeForce RTX 3060 Ti, 8192MiB)
YOLOv8s-seg summary (fused): 86 layers, 11,780,374 parameters, 0 gradients, 39.9 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1/1 7.7it/s 0.1s
                   all         12         24      0.589      0.792      0.702      0.302      0.862       0.88      0.946      0.607
          divider_left         12         12      0.627       0.75       0.68      0.227       0.91      0.844      0.924      0.538
         divider_right         12         12      0.552      0.833      0.723      0.377      0.814      0.917      0.968      0.676

```


### What do these numbers actually mean?
- **mAP50 (94.6%):** This means that in 94.6% of cases, the AI's predicted "blob" for a divider overlapped with our manual label by at least 50%. For a long, thin divider line, this is extremely high accuracy.
- **Class Divide:** Interestingly, `divider_right` (0.968 mAP) outperformed `divider_left` (0.924). We realized the "English" column on the right has very distinct, consistent whitespace compared to the Ge'ez/Amharic gutters, making it easier for the model to "see."

---

## 5. The "Rectangular Trap" and the In-House Tool
Even with a 94% accurate model, we encountered a final hurdle: **Deployment Logic.**
Surya and other tools wanted "Rectangular Crops." But our dividers were tilted. If we took a rectangular crop of a tilted column, we'd lose the corners of the text.

We tried several methods:
1. **Centroid Slicing:** Take the center of the AI's mask. (Failed on diagonal skews).
2. **Surya-Native:** Let Surya guess. (Merged columns again).

**The Decision:** We rejected the shims. We decided to build our own **In-House HITL Line Editor**.
We didn't want to just "label" more—we wanted to "verify."
We built a FastAPI web tool with a custom SQLite backend. Why SQLite? Because we needed ACID-compliant state management to ensure that if the server crashed during your 4-hour review session, not a single coordinate was lost.

### The Geometry Secret: The "Vector Pivot"
In our tool, we moved away from "drawing boxes." We implemented **Least-Squares Linear Regression** (`cv2.fitLine`). We took the AI's messy mask and calculated a **Vector Trajectory**.
In the UI, you were manipulating **Vectors**, not lines. This is why you could grab the top handle and pivot the line—it followed the actual physics of the document scan.

---

## 6. The "Golden State" and Why we Ignored "Data"
As we finalized, we made a hard architectural choice for the(`.gitignore`):
**We blocked all raw data, ZIPs, and intermediate JSON dumps.**
**Why?** Because to "templatize" this for the next researcher, we don't want to give them 10GB of raw JPEGs. We want to give them the **Logic** and the **Weights**.
- We tracked `models/layout/weights/berana_yolov8_divider_v13.pt`.
- We tracked `verified_lines_state.json`.

Now, anyone can pull this repo, and they have the "Brain" (the model) and the "Soul" (your verified coordinates). They can skip the 48 hours of labeling we did and go straight to OCR.

---

## 7. Retrospective: Lessons from the Gutter
If there is one thing we learned, it's that **Layout is a Geometry Problem, not an Image Problem.**
- We learned that **Surya** is a king of reading characters but a peasant at understanding columns.
- We learned that **Labels** are only as good as the tool used to verify them.
- We learned that **Affine Warping** (the next phase) is the only way to truly "fix" a tilted document before OCR.

We started with a crooked PDF and finished with a precision-mapped coordinate system. We didn't just automate a process; we mastered the geometry of the page.

---
*Documented by the Berana Development Team - Feb 2026*
