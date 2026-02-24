# Berana Custom Model Training Guide (Optional)

This guide is **strictly for developers and researchers** who need to train a YOLOv8-Segmentation layout model from scratch on a new type of manuscript.

If you are only running OCR on ancient Ethiopic manuscripts similar to our baseline, **you do not need to do this**. You can use the provided `berana_yolov8_divider_v13.pt` weights and skip entirely to Phase 1 (Layout Inference) in the main pipeline.

---

## 1. Setup Label Studio (The Annotation Environment)

First, initialize the local Docker environment for Label Studio. We decouple this from the main `setup.sh` to keep your base environment clean.

```bash
cd tools/label_studio
./setup_label_studio.sh
docker compose up -d
```

Navigate to `http://localhost:8080`.
The default login created by docker-compose is typically defined in your `.env` (check `tools/label_studio/.env` for your exact `LABEL_STUDIO_USERNAME` and `PASSWORD`).

## 1.1 Exact Label Studio Workflow (Import/Export Contract)

Use this exact flow to avoid path/import mismatches:

1. Open Label Studio at `http://localhost:8080` and create/select your project.
2. In project settings, set the labeling Setup to --> Custom Template --> Code = `tools/label_studio/project_ui.xml`.
3. Configure **Cloud Storage -> Local Files**:
   - Storage type: `Local files`
   - Absolute local path inside container: `/label-studio/files/visuals`
   - Enable synchronization.
4. Run auto-label generation from project root:
   - `python berana.py layout-infer --pdf-path data/raw_pdfs/doc_001.Triple.pdf`
   - Default JSON output: `output/layout_auto/<doc_stem>_vNN/auto_labels_tasks.json`
5. Import predictions/tasks:
   - In Label Studio project: `Import` -> `Upload Files`
   - Select the generated JSON file (`output/layout_auto/<doc_stem>_vNN/auto_labels_tasks.json`)
   - Import type: **JSON task file** (not CSV, not raw images).
6. Human review:
   - Validate/fix polygons for `divider_left` and `divider_right`.
7. Export for training:
   - `Export` -> format: **YOLO (Polygon)**
   - Choose **YOLO (without images)** to avoid duplicate image payloads and keep repo workflows lean.
   - Place resulting ZIP under `data/layout_dataset/` for ingestion/training scripts.

## 2. Ingest "Unseen" Baseline Data

Use the main pipeline to chop your raw PDF into a format Label Studio can digest, but use the `--slice-only` flag so it skips OCR extraction.

```bash
# From the project root path
source .venv/bin/activate
python berana.py ingest --pdf-path data/raw_pdfs/my_custom_manuscript.pdf --slice-only
```

Use current pagination flags:
```bash
python berana.py ingest \
  --pdf-path data/raw_pdfs/my_custom_manuscript.pdf \
  --slice-only \
  --start-page 1 \
  --end-page 30
```

This command generates images into a mounted `berana_data` folder and creates an auto-import JSON. Follow standard Label Studio import procedures to map these tasks into your new project.

## 3. Labeling the "Golden Target"

Inside Label Studio:
1. Select the **Polygon** tool.
2. Carefully draw masks representing the physical "gutters" (the whitespace separating your columns).
3. Assign them the exact classes expected by the pipeline (e.g., `divider_left`, `divider_right`).
4. Keep the labels consistent on overlapping text, ignoring minor fading or bleed-through.

Label at least **30 distinct layout pages** to acquire enough geometric variance.

## 4. Export & Prep the YOLO Dataset

Once manual labeling is complete, export the project from Label Studio in the **YOLO (Polygon)** format. Move the `.zip` file into `data/layout_dataset/`.

Next, run our data ingestion tool to format the raw YOLO export into a strict ultrasonic-compliant training schema:

```bash
python tools/ingest_labels.py \
    --zip "data/layout_dataset/project-export.zip" \
    --images-dir "data/layout_dataset/images" \
    --output "data/layout_dataset/yolo_train_v1"
```

## 5. Train the Model

Now you will execute the training using the YOLOv8-Segmentation backend with heavy Mosaic Augmentation (crucial for partial-page visibility).

```bash
# Make sure your VENV is active
python berana.py train-layout \
    --data-yaml "data/layout_dataset/yolo_train_v1/dataset.yaml" \
    --epochs 100 \
    --img-size 1024 \
    --batch-size 16 \
    --name "my_custom_divider_v1"
```

*Note: You may need to tune `--batch-size` down to `8` or `4` if you encounter CUDA Out-Of-Memory (OOM) errors on GPUs with less than 8GB of VRAM.*

## 6. The "Active Learning" Loop

The most efficient way to scale from 30 pages to 500 pages of ground truth is **Active Learning**:

1. After Training v1 completes, its weights are placed in `runs/segment/runs/layout/my_custom_divider_v1/weights/best.pt`.
2. Move those weights into `models/layout/weights/`.
3. Run `python berana.py layout-infer` on another 50 unlabelled pages.
   - This generates auto-label tasks at `output/layout_auto/<doc_stem>_vNN/auto_labels_tasks.json` by default.
4. Import those predictions back into Label Studio.
5. Manually correct the AI's mistakes (which takes 10x less time than drawing polygons from scratch).
6. Re-export and train v2.

By iteration v3 or v4, the model will reach `~0.95 mAP50`, and you can graduate to the standard HITL Line Editor phase defined in the main `README.md`.
