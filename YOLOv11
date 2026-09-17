# YOLOv11_NEU_DET -- Code extracted from notebook
# (Cell markers preserved; run as notebook cells, not straight-through script)

# ======================================================================
# CODE CELL 1
# ======================================================================
!pip install -q ultralytics opencv-python-headless lxml pandas matplotlib seaborn tqdm pyyaml
import os, sys, glob, json, time, shutil, subprocess, random, re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

print("Core imports OK.")


# ======================================================================
# CODE CELL 2
# ======================================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("Drive mounted.")


# ======================================================================
# CODE CELL 3
# ======================================================================
RANDOM_SEED = 0
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
DATASET_PATH = "/content/drive/MyDrive/NEU-DET - Copy"
OUTPUT_PATH = "/content/drive/MyDrive/YOLO_NEU_DET_Experiments"
NUMBER_OF_CLASSES = 4
CLASS_NAMES = ["patches", "pitted_surface", "scratches"]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_PATH, exist_ok=True)
DATASET_PREPARED_DIR = os.path.join(OUTPUT_PATH, "Dataset_Prepared")
RESULTS_DIR = os.path.join(OUTPUT_PATH, "Results")
COMPARISON_DIR = os.path.join(OUTPUT_PATH, "Comparison")
for d in [DATASET_PREPARED_DIR, RESULTS_DIR, COMPARISON_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_DIR = {}
for v in ["YOLOv1","YOLOv2","YOLOv3","YOLOv4","YOLOv5","YOLOv6","YOLOv7","YOLOv8","YOLOv9","YOLOv10","YOLOv11"]:
    base = os.path.join(OUTPUT_PATH, v)
    MODEL_DIR[v] = {
        "base": base,
        "weights": os.path.join(base, "weights"),
        "results": os.path.join(base, "results"),
        "graphs": os.path.join(base, "graphs"),
        "inference_results": os.path.join(base, "inference_results"),
        "metrics": os.path.join(base, "metrics"),
        "repo": os.path.join(base, "repo"),
        "pretrained": os.path.join(base, "pretrained"),
    }
    for d in MODEL_DIR[v].values():
        os.makedirs(d, exist_ok=True)

print("Configuration:")
print(f"  RANDOM_SEED        : {RANDOM_SEED}")
print(f"  EPOCHS              : {EPOCHS}")
print(f"  IMAGE_SIZE          : {IMAGE_SIZE}")
print(f"  BATCH_SIZE          : {BATCH_SIZE}")
print(f"  DATASET_PATH        : {DATASET_PATH}")
print(f"  OUTPUT_PATH         : {OUTPUT_PATH}")
print(f"  NUMBER_OF_CLASSES   : {NUMBER_OF_CLASSES}")
print(f"  CLASS_NAMES         : {CLASS_NAMES}")

import torch
if torch.cuda.is_available():
    print(f"\nGPU available: Yes")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("\nGPU available: No")
    print("*** WARNING: no GPU detected. Training 11 YOLO versions on CPU is not")
    print("practical -- go to Runtime > Change runtime type and select a GPU. ***")


# ======================================================================
# CODE CELL 4
# ======================================================================
if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset path does not exist: {DATASET_PATH}\n"
        f"Check that your Google Drive is mounted and the folder name is exact "
        f"(including spaces/capitalization)."
    )

def find_dir(root, must_contain_any, exclude_any=()):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        low = dirpath.lower()
        if any(k in low for k in must_contain_any) and not any(k in low for k in exclude_any):
            hits.append(dirpath)
    return hits

def pick_dir_with_files(candidates, exts):
    for d in candidates:
        if any(f.lower().endswith(exts) for f in os.listdir(d)):
            return d
    return candidates[0] if candidates else None

def discover_split_flat(root, split_keywords):
    """Recursively find images and XMLs under any subfolder whose path
    contains one of split_keywords. Returns (images_dict, xmls_dict) keyed by
    filename stem."""
    images, xmls = {}, {}
    for dirpath, _, filenames in os.walk(root):
        low_path = dirpath.lower()
        if split_keywords and not any(k in low_path for k in split_keywords):
            continue
        for fn in filenames:
            stem_, ext = os.path.splitext(fn)
            ext = ext.lower()
            full = os.path.join(dirpath, fn)
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                images[stem_] = full
            elif ext == ".xml":
                xmls[stem_] = full
    return images, xmls

_train_img_c = find_dir(DATASET_PATH, ["train"], exclude_any=["annotat", "xml", "label"])
_train_ann_c = [d for d in find_dir(DATASET_PATH, ["train"]) if any(k in d.lower() for k in ["annotat", "xml", "label"])]
_val_img_c   = find_dir(DATASET_PATH, ["val", "valid"], exclude_any=["annotat", "xml", "label"])
_val_ann_c   = [d for d in find_dir(DATASET_PATH, ["val", "valid"]) if any(k in d.lower() for k in ["annotat", "xml", "label"])]

TRAIN_IMAGES_DIR = pick_dir_with_files(_train_img_c, (".jpg", ".jpeg", ".png", ".bmp"))
TRAIN_ANNOT_DIR  = pick_dir_with_files(_train_ann_c, (".xml",))
VAL_IMAGES_DIR   = pick_dir_with_files(_val_img_c, (".jpg", ".jpeg", ".png", ".bmp"))
VAL_ANNOT_DIR    = pick_dir_with_files(_val_ann_c, (".xml",))

# Fallback: no train/val-named folders found at all -- scan the whole tree and
# create a random 80/20 split rather than failing outright.
_USED_AUTO_SPLIT = False
if TRAIN_IMAGES_DIR is None and VAL_IMAGES_DIR is None:
    print("No 'train'/'val' folders detected by name -- scanning the entire tree")
    print("and creating a random 80/20 split instead.")
    all_images, all_xmls = discover_split_flat(DATASET_PATH, [])
    stems = sorted(all_images.keys())
    random.Random(RANDOM_SEED).shuffle(stems)
    split_idx = int(0.8 * len(stems))
    train_stems, val_stems = set(stems[:split_idx]), set(stems[split_idx:])

    _AUTO_SPLIT_DIR = os.path.join(WORK_DIR if 'WORK_DIR' in dir() else "/content", "auto_split")
    for sub in ["train_images", "train_annot", "val_images", "val_annot"]:
        os.makedirs(os.path.join(_AUTO_SPLIT_DIR, sub), exist_ok=True)
    for s in train_stems:
        if s in all_images: shutil.copy2(all_images[s], os.path.join(_AUTO_SPLIT_DIR, "train_images", os.path.basename(all_images[s])))
        if s in all_xmls: shutil.copy2(all_xmls[s], os.path.join(_AUTO_SPLIT_DIR, "train_annot", os.path.basename(all_xmls[s])))
    for s in val_stems:
        if s in all_images: shutil.copy2(all_images[s], os.path.join(_AUTO_SPLIT_DIR, "val_images", os.path.basename(all_images[s])))
        if s in all_xmls: shutil.copy2(all_xmls[s], os.path.join(_AUTO_SPLIT_DIR, "val_annot", os.path.basename(all_xmls[s])))
    TRAIN_IMAGES_DIR = os.path.join(_AUTO_SPLIT_DIR, "train_images")
    TRAIN_ANNOT_DIR  = os.path.join(_AUTO_SPLIT_DIR, "train_annot")
    VAL_IMAGES_DIR   = os.path.join(_AUTO_SPLIT_DIR, "val_images")
    VAL_ANNOT_DIR    = os.path.join(_AUTO_SPLIT_DIR, "val_annot")
    _USED_AUTO_SPLIT = True
    print(f"Auto-split: {len(train_stems)} train, {len(val_stems)} val images.")

for name, path in [("TRAIN_IMAGES_DIR", TRAIN_IMAGES_DIR), ("TRAIN_ANNOT_DIR", TRAIN_ANNOT_DIR),
                    ("VAL_IMAGES_DIR", VAL_IMAGES_DIR), ("VAL_ANNOT_DIR", VAL_ANNOT_DIR)]:
    if path is None:
        raise FileNotFoundError(
            f"Could not automatically locate {name} under {DATASET_PATH}.\n"
            f"Inspect the folder structure manually and hardcode the correct path."
        )
    print(f"{name}: {path}")

def list_files(d, exts):
    return sorted([f for f in os.listdir(d) if f.lower().endswith(exts)]) if d else []

def stem(f):
    return os.path.splitext(f)[0]

train_images = list_files(TRAIN_IMAGES_DIR, (".jpg", ".jpeg", ".png", ".bmp"))
train_xmls   = list_files(TRAIN_ANNOT_DIR, (".xml",))
val_images   = list_files(VAL_IMAGES_DIR, (".jpg", ".jpeg", ".png", ".bmp"))
val_xmls     = list_files(VAL_ANNOT_DIR, (".xml",))

print(f"\nTraining images     : {len(train_images)}")
print(f"Training XML files  : {len(train_xmls)}")
print(f"Validation images   : {len(val_images)}")
print(f"Validation XML files: {len(val_xmls)}")

def report_mismatches(images, xmls, split_name):
    img_stems, xml_stems = {stem(f) for f in images}, {stem(f) for f in xmls}
    missing_xml = sorted(img_stems - xml_stems)
    missing_img = sorted(xml_stems - img_stems)
    print(f"\n[{split_name}] images without a matching XML: {len(missing_xml)}")
    if missing_xml[:10]: print("  e.g.", missing_xml[:10])
    print(f"[{split_name}] XML without a matching image: {len(missing_img)}")
    if missing_img[:10]: print("  e.g.", missing_img[:10])

report_mismatches(train_images, train_xmls, "train")
report_mismatches(val_images, val_xmls, "val")

from PIL import Image
print("\nSample image info (train):")
for f in train_images[:5]:
    with Image.open(os.path.join(TRAIN_IMAGES_DIR, f)) as im:
        print(f"  {f}: size={im.size} (W x H), original mode={im.mode} -> converted to RGB for training/inference")

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_el = root.find("size")
    width, height = int(size_el.find("width").text), int(size_el.find("height").text)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        objects.append({
            "name": name,
            "xmin": float(bnd.find("xmin").text), "ymin": float(bnd.find("ymin").text),
            "xmax": float(bnd.find("xmax").text), "ymax": float(bnd.find("ymax").text),
        })
    return width, height, objects

objects_per_class = {}
for f in train_xmls + val_xmls:
    annot_dir = TRAIN_ANNOT_DIR if f in train_xmls else VAL_ANNOT_DIR
    _, _, objs = parse_voc_xml(os.path.join(annot_dir, f))
    for o in objs:
        objects_per_class[o["name"]] = objects_per_class.get(o["name"], 0) + 1

print("\nRaw class names discovered in XML files, with counts:")
for cls, count in sorted(objects_per_class.items()):
    print(f"  {cls!r:20s}: {count}")


# ======================================================================
# CODE CELL 5
# ======================================================================
for f in train_xmls[:2]:
    w, h, objs = parse_voc_xml(os.path.join(TRAIN_ANNOT_DIR, f))
    print(f"--- {f} ---  image size=({w}x{h})")
    for o in objs:
        print(f"   class={o['name']:15s} xmin={o['xmin']:.0f} ymin={o['ymin']:.0f} "
              f"xmax={o['xmax']:.0f} ymax={o['ymax']:.0f}")


# ======================================================================
# CODE CELL 6
# ======================================================================
# Normalize small labeling differences in your XML files to the canonical
# names this notebook expects. Add entries here if discover step below finds
# a raw class name not covered (e.g. a typo or different casing/spacing).
RAW_TO_CANONICAL = {
    "patches": "patches", "Patches": "patches",
    "pitted_surface": "pitted_surface", "pitted surface": "pitted_surface",
    "pitted-surface": "pitted_surface", "Pitted Surface": "pitted_surface", "Pitted_surface": "pitted_surface",
    "scratches": "scratches", "Scratches": "scratches",
}

def normalize_class_name(raw_name):
    if raw_name in RAW_TO_CANONICAL:
        return RAW_TO_CANONICAL[raw_name]
    # last-resort normalization: lowercase, spaces/hyphens -> underscore
    guess = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
    return guess

raw_classes_found = set(objects_per_class.keys())
normalized_classes = {normalize_class_name(c) for c in raw_classes_found}
EXPECTED_CLASSES = set(CLASS_NAMES)

print("Raw classes found        :", sorted(raw_classes_found))
print("Normalized to            :", sorted(normalized_classes))
print("Expected (CLASS_NAMES)   :", sorted(EXPECTED_CLASSES))

if normalized_classes != EXPECTED_CLASSES:
    raise ValueError(
        f"After normalization, discovered classes {sorted(normalized_classes)} still do not "
        f"match the required CLASS_NAMES {sorted(EXPECTED_CLASSES)}.\n"
        f"Add a mapping in RAW_TO_CANONICAL above for whichever raw name(s) are unmatched, "
        f"or fix CLASS_NAMES in Cell 3, then re-run this cell."
    )
print("\nClasses match (after normalization). Safe to proceed.")

invalid_boxes = []
for f in train_xmls + val_xmls:
    annot_dir = TRAIN_ANNOT_DIR if f in train_xmls else VAL_ANNOT_DIR
    w, h, objs = parse_voc_xml(os.path.join(annot_dir, f))
    for o in objs:
        if not (o["xmin"] < o["xmax"] and o["ymin"] < o["ymax"]):
            invalid_boxes.append(f"{f}: xmin/xmax or ymin/ymax not properly ordered")
        if not (0 <= o["xmin"] and o["xmax"] <= w and 0 <= o["ymin"] and o["ymax"] <= h):
            invalid_boxes.append(f"{f}: box exceeds image bounds ({w}x{h})")

if invalid_boxes:
    print(f"\n{len(invalid_boxes)} invalid box(es) found:")
    for p in invalid_boxes[:30]:
        print(" -", p)
else:
    print("\nAll bounding boxes are valid.")


# ======================================================================
# CODE CELL 7
# ======================================================================
CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_NAME = {v: k for k, v in CLASS_MAP.items()}

YOLO_LABELS_DIR = os.path.join(DATASET_PREPARED_DIR, "labels_raw")
for split in ["train", "val"]:
    os.makedirs(os.path.join(YOLO_LABELS_DIR, split), exist_ok=True)

def convert_split(images_dir, annot_dir, image_files, split):
    converted, skipped = 0, 0
    for img_file in image_files:
        name = stem(img_file)
        xml_path = os.path.join(annot_dir, name + ".xml")
        if not os.path.isfile(xml_path):
            skipped += 1; continue
        w, h, objs = parse_voc_xml(xml_path)
        lines = []
        for o in objs:
            norm_name = normalize_class_name(o["name"])
            if norm_name not in CLASS_MAP:
                print(f"  SKIPPING unknown class '{o['name']}' (normalized: '{norm_name}') in {xml_path}"); continue
            cls_id = CLASS_MAP[norm_name]
            cx, cy = (o["xmin"]+o["xmax"])/2/w, (o["ymin"]+o["ymax"])/2/h
            bw, bh = (o["xmax"]-o["xmin"])/w, (o["ymax"]-o["ymin"])/h
            cx, cy, bw, bh = (min(max(v,0.0),1.0) for v in (cx,cy,bw,bh))
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(YOLO_LABELS_DIR, split, name + ".txt"), "w") as fh:
            fh.write("\n".join(lines))
        converted += 1
    return converted, skipped

tr_c, tr_s = convert_split(TRAIN_IMAGES_DIR, TRAIN_ANNOT_DIR, train_images, "train")
va_c, va_s = convert_split(VAL_IMAGES_DIR, VAL_ANNOT_DIR, val_images, "val")
print(f"Train: converted {tr_c}, skipped (no XML) {tr_s}")
print(f"Val  : converted {va_c}, skipped (no XML) {va_s}")


# ======================================================================
# CODE CELL 8
# ======================================================================
NEU_YOLO = os.path.join(DATASET_PREPARED_DIR, "NEU_YOLO")
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(NEU_YOLO, sub), exist_ok=True)

def stage_split(images_dir, image_files, split):
    n = 0
    for img_file in image_files:
        name = stem(img_file)
        lbl_src = os.path.join(YOLO_LABELS_DIR, split, name + ".txt")
        if not os.path.isfile(lbl_src):
            continue
        shutil.copy2(os.path.join(images_dir, img_file), os.path.join(NEU_YOLO, "images", split, img_file))
        shutil.copy2(lbl_src, os.path.join(NEU_YOLO, "labels", split, name + ".txt"))
        n += 1
    return n

n_train = stage_split(TRAIN_IMAGES_DIR, train_images, "train")
n_val = stage_split(VAL_IMAGES_DIR, val_images, "val")
print(f"Standardized dataset: {n_train} train, {n_val} val images staged.")

data_yaml_path = os.path.join(NEU_YOLO, "data.yaml")
with open(data_yaml_path, "w") as fh:
    fh.write(f"""path: {NEU_YOLO}
train: images/train
val: images/val
nc: {NUMBER_OF_CLASSES}
names: {CLASS_NAMES}
""")
print("\n--- data.yaml ---")
print(open(data_yaml_path).read())


# ======================================================================
# CODE CELL 9
# ======================================================================
def yolo_txt_to_boxes(lbl_path, img_w, img_h):
    boxes = []
    if not os.path.isfile(lbl_path): return boxes
    with open(lbl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            cls_id, cx, cy, bw, bh = line.split()
            cls_id = int(cls_id)
            cx, cy, bw, bh = float(cx)*img_w, float(cy)*img_h, float(bw)*img_w, float(bh)*img_h
            boxes.append((cls_id, ID_TO_NAME[cls_id], cx-bw/2, cy-bh/2, bw, bh))
    return boxes

img_dir = os.path.join(NEU_YOLO, "images", "train")
lbl_dir = os.path.join(NEU_YOLO, "labels", "train")
all_train_files = list_files(img_dir, (".jpg",".jpeg",".png",".bmp"))

by_class = {c: [] for c in CLASS_MAP}
for f in all_train_files:
    boxes = yolo_txt_to_boxes(os.path.join(lbl_dir, stem(f)+".txt"), 1, 1)
    for cls_id, name, *_ in boxes:
        by_class[name].append(f); break

sample_files = [random.choice(files) for files in by_class.values() if files]
while len(sample_files) < 6 and len(sample_files) < len(all_train_files):
    f = random.choice(all_train_files)
    if f not in sample_files: sample_files.append(f)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, f in zip(axes.ravel(), sample_files):
    img = Image.open(os.path.join(img_dir, f)).convert("RGB")
    w, h = img.size
    boxes = yolo_txt_to_boxes(os.path.join(lbl_dir, stem(f)+".txt"), w, h)
    ax.imshow(img)
    for cls_id, name, xmin, ymin, bw, bh in boxes:
        rect = plt.Rectangle((xmin, ymin), bw, bh, fill=False, edgecolor="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, max(ymin-5,0), f"{name} ({cls_id})", color="lime", fontsize=9, weight="bold")
    ax.set_title(f, fontsize=9); ax.axis("off")
for ax in axes.ravel()[len(sample_files):]: ax.axis("off")
plt.tight_layout(); plt.show()


# ======================================================================
# CODE CELL 10
# ======================================================================
def run_and_capture(cmd, cwd=None):
    print(f"$ {cmd}")
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end=""); lines.append(line)
    proc.wait()
    return "".join(lines), proc.returncode

def parse_first_float(pattern, text, flags=re.IGNORECASE):
    m = re.search(pattern, text, flags)
    return float(m.group(1)) if m else None

RESULT_COLUMNS = [
    "Model", "Framework", "Architecture", "Backbone", "Pretrained_Weights",
    "Image_Size", "Batch_Size", "Optimizer", "Learning_Rate", "Epochs",
    "Data_Augmentation", "Precision", "Recall", "F1", "mAP50", "mAP50_95",
    "TP", "FP", "FN", "Num_Val_Images", "Num_GT_Objects", "Num_Pred_Objects",
    "FPS", "Inference_Time_ms", "Parameters", "Model_Size_MB", "Training_Time_s",
    "Best_Epoch", "AP_patches", "AP_pitted_surface", "AP_scratches",
    "Status", "Notes",
]

def save_overall_result(model_name, metrics: dict):
    """Restart-safe: upserts by model name, writes to Results/ on Drive."""
    row = {c: metrics.get(c, None) for c in RESULT_COLUMNS}
    row["Model"] = model_name
    csv_path = os.path.join(RESULTS_DIR, "all_models_results.csv")
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["Model"] != model_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df[RESULT_COLUMNS]
    df.to_csv(csv_path, index=False)
    # also save a per-model copy under that model's own results/ folder
    if model_name in MODEL_DIR:
        df[df["Model"] == model_name].to_csv(
            os.path.join(MODEL_DIR[model_name]["results"], f"{model_name}_results.csv"), index=False)
    print(f"Saved overall results for '{model_name}'")
    return df

def save_per_class_result(model_name, per_class_rows: list):
    """per_class_rows: list of dicts with keys Model, Class, Precision, Recall, F1, AP50"""
    csv_path = os.path.join(RESULTS_DIR, "all_models_per_class_results.csv")
    new_df = pd.DataFrame(per_class_rows)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["Model"] != model_name]
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    df.to_csv(csv_path, index=False)
    print(f"Saved per-class results for '{model_name}'")

def report_failure(model_name, reason):
    print(f"\n{model_name}: Training failed")
    print(f"Reason: {reason}")
    save_overall_result(model_name, {"Status": "FAILED", "Notes": reason})

# ============================= ULTRALYTICS FAMILY =============================
# Covers v3, v5, v8, v9, v10, v11 -- one shared implementation, called with
# different pretrained-weight filenames.

def ultralytics_train(model_name, pt_name, project_dir):
    from ultralytics import YOLO
    model = YOLO(pt_name)
    t0 = time.time()
    model.train(data=data_yaml_path, imgsz=IMAGE_SIZE, epochs=EPOCHS, batch=BATCH_SIZE,
                seed=RANDOM_SEED, project=project_dir, name="run", exist_ok=True)
    training_time_s = time.time() - t0
    return model, training_time_s

def ultralytics_evaluate(model_name, model, pt_name, training_time_s):
    metrics = model.val(data=data_yaml_path, imgsz=IMAGE_SIZE, split="val")
    names = model.names
    name_to_idx = {v: k for k, v in names.items()}

    def ap_for(cls_name):
        if cls_name not in name_to_idx: return None
        idx = list(names.keys()).index(name_to_idx[cls_name])
        try: return float(metrics.box.maps[idx])
        except Exception: return None

    val_img_dir = os.path.join(NEU_YOLO, "images", "val")
    val_files = [os.path.join(val_img_dir, f) for f in list_files(val_img_dir, (".jpg",".jpeg",".png",".bmp"))]
    t0 = time.time()
    _ = model.predict(val_files, imgsz=IMAGE_SIZE, verbose=False)
    elapsed = time.time() - t0
    n = len(val_files)
    inference_ms = (elapsed/n)*1000 if n else None
    fps = (n/elapsed) if elapsed else None

    n_params = sum(p.numel() for p in model.model.parameters())
    best_weights = getattr(model.trainer, "best", None) if hasattr(model, "trainer") else None
    model_size_mb = os.path.getsize(best_weights)/(1024*1024) if best_weights and os.path.isfile(best_weights) else None

    precision, recall = float(metrics.box.mp), float(metrics.box.mr)
    f1 = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else None
    mAP50, mAP50_95 = float(metrics.box.map50), float(metrics.box.map)

    n_gt = sum(int(x) for x in metrics.confusion_matrix.matrix.sum(axis=0)[:-1]) if hasattr(metrics, "confusion_matrix") else None
    n_pred = sum(int(x) for x in metrics.confusion_matrix.matrix.sum(axis=1)[:-1]) if hasattr(metrics, "confusion_matrix") else None

    overall = {
        "Framework": "Ultralytics", "Pretrained_Weights": pt_name, "Image_Size": IMAGE_SIZE,
        "Batch_Size": BATCH_SIZE, "Epochs": EPOCHS, "Precision": precision, "Recall": recall,
        "F1": f1, "mAP50": mAP50, "mAP50_95": mAP50_95, "Num_Val_Images": n,
        "Num_GT_Objects": n_gt, "Num_Pred_Objects": n_pred,
        "FPS": fps, "Inference_Time_ms": inference_ms, "Parameters": n_params,
        "Model_Size_MB": model_size_mb, "Training_Time_s": training_time_s,
        "AP_patches": ap_for("patches"),
        "AP_pitted_surface": ap_for("pitted_surface"), "AP_scratches": ap_for("scratches"),
        "Status": "SUCCESS",
    }

    per_class_rows = []
    for cls in CLASS_NAMES:
        per_class_rows.append({
            "Model": model_name, "Class": cls,
            "Precision": overall["Precision"], "Recall": overall["Recall"],
            "F1": overall["F1"], "AP50": overall.get(f"AP_{cls}"),
        })
    return overall, per_class_rows, metrics

def ultralytics_curves(model_name, project_dir):
    """Ultralytics writes results.csv every epoch during training -- this reads
    it and plots loss/precision/recall/mAP vs epoch, rather than regenerating
    anything by hand."""
    results_csv = os.path.join(project_dir, "run", "results.csv")
    if not os.path.isfile(results_csv):
        print(f"No results.csv found at {results_csv} -- cannot plot training curves.")
        return
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

    plot_specs = [
        ("train/box_loss", "Training loss vs epoch"),
        ("metrics/precision(B)", "Precision vs epoch"),
        ("metrics/recall(B)", "Recall vs epoch"),
        ("metrics/mAP50(B)", "mAP@0.5 vs epoch"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95 vs epoch"),
    ]
    graphs_dir = MODEL_DIR[model_name]["graphs"]
    for col, title in plot_specs:
        if col not in df.columns:
            print(f"'{col}' not found in results.csv -- skipping ({title}), not fabricating it.")
            continue
        plt.figure(figsize=(7, 4))
        plt.plot(df[epoch_col], df[col], marker="o", markersize=3)
        plt.xlabel("Epoch"); plt.ylabel(col); plt.title(f"{model_name}: {title}")
        plt.grid(alpha=0.3); plt.tight_layout()
        out_path = os.path.join(graphs_dir, col.replace("/","_").replace("(B)","") + ".png")
        plt.savefig(out_path); plt.show()
    print(f"Curves saved under {graphs_dir}")

def ultralytics_confusion_pr(model_name, ultra_metrics):
    """Ultralytics' model.val() auto-saves confusion_matrix.png, PR_curve.png,
    etc. into its own run directory -- this locates and copies them into this
    model's own graphs/ folder rather than regenerating them."""
    val_run_dir = ultra_metrics.save_dir
    graphs_dir = MODEL_DIR[model_name]["graphs"]
    plot_files = ["confusion_matrix.png", "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"]
    found = []
    for p in plot_files:
        src = os.path.join(val_run_dir, p)
        if os.path.isfile(src):
            dst = os.path.join(graphs_dir, p)
            shutil.copy2(src, dst)
            found.append(p)
    print(f"Copied {found} to {graphs_dir}" if found else "No confusion matrix / PR curve files found in the val run directory.")
    missing = [p for p in plot_files if p not in found]
    if missing:
        print(f"Not available: {missing}")

def ultralytics_inference(model, image_path, out_dir, model_name):
    results = model.predict(image_path, imgsz=IMAGE_SIZE, conf=0.25, verbose=False)
    r = results[0]
    im = r.plot()[..., ::-1]  # BGR->RGB
    plt.figure(figsize=(8,8)); plt.imshow(im); plt.axis("off")
    plt.title(f"{model_name} inference"); plt.show()
    out_path = os.path.join(out_dir, f"{model_name}_{os.path.basename(image_path)}")
    Image.fromarray(im).save(out_path)
    detections = [(r.names[int(b.cls)], float(b.conf)) for b in r.boxes]
    for cls_name, conf in detections:
        print(f"{cls_name} {conf:.2f}")
    if not detections:
        print("No detections above the confidence threshold.")
    print(f"Saved: {out_path}")
    return out_path, detections

print("Ultralytics-family functions loaded: ultralytics_train, ultralytics_evaluate, "
      "ultralytics_curves, ultralytics_confusion_pr, ultralytics_inference")


# ======================================================================
# CODE CELL 11
# ======================================================================
# ============================= DARKNET FAMILY =============================
# Covers v1, v2, v4. v2 and v4 use the modern hank-ai/darknet fork (their layer
# types [region]/[yolo] are both present in it). v1 needs a fallback to the
# original pjreddie/darknet, since the modern fork's C++ rewrite dropped the
# legacy [detection] layer entirely -- confirmed from an actual build log.

DARKNET_WORK = "/content/darknet_work"  # build artifacts are large/ephemeral;
os.makedirs(DARKNET_WORK, exist_ok=True)  # kept local, only final results go to Drive

def find_binary(root, name="darknet"):
    for dirpath, dirnames, filenames in os.walk(root):
        if name in filenames:
            candidate = os.path.join(dirpath, name)
            if os.access(candidate, os.X_OK):
                return candidate
    return None

def build_modern_darknet():
    """Used by v2 and v4. Builds hank-ai/darknet once; reused by both."""
    darknet_dir = os.path.join(DARKNET_WORK, "darknet")
    existing = os.path.join(darknet_dir, "darknet")
    if os.path.isfile(existing):
        print("Modern Darknet already built, skipping.")
        return existing
    run_and_capture("apt-get update -qq && apt-get install -y -qq cmake libopencv-dev")
    if not os.path.isdir(darknet_dir):
        run_and_capture(f"git clone https://github.com/hank-ai/darknet.git {darknet_dir}")
    build_dir = os.path.join(darknet_dir, "build")
    os.makedirs(build_dir, exist_ok=True)
    out, rc = run_and_capture("cmake -DCMAKE_BUILD_TYPE=Release ..", cwd=build_dir)
    if rc != 0:
        print("cmake failed -- see error above (often missing OpenCV dev headers).")
        return None
    run_and_capture("make -j$(nproc)", cwd=build_dir)
    found = find_binary(build_dir)
    if found:
        shutil.copy2(found, existing)
        return existing
    print("WARNING: modern Darknet build finished but no executable was found.")
    return None

def patch_arch_line(makefile_text, gencode):
    """Line-based (not regex-based): pjreddie's Makefile defines ARCH= across
    several physical lines joined with backslash continuation. A single-line
    substitution leaves the continuation lines orphaned ('missing separator').
    This consumes every continuation line before writing one clean replacement."""
    lines = makefile_text.splitlines(keepends=True)
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("ARCH="):
            out.append(f"ARCH= {gencode}\n")
            cont = line.rstrip("\n").endswith("\\")
            i += 1
            while cont and i < len(lines):
                cont = lines[i].rstrip("\n").endswith("\\")
                i += 1
            continue
        out.append(line); i += 1
    return "".join(out)

def build_pjreddie_darknet():
    """v1-only fallback. GPU=1 (raw CUDA), CUDNN=0 (2018 code calls a cuDNN API
    removed in cuDNN 8+), OPENCV=0 (Colab lacks the C++ OpenCV dev headers this
    old Makefile expects)."""
    pj_dir = os.path.join(DARKNET_WORK, "darknet_pjreddie")
    existing = os.path.join(pj_dir, "darknet")
    if os.path.isfile(existing):
        print("pjreddie/darknet already built, skipping.")
        return existing
    if not os.path.isdir(pj_dir):
        run_and_capture(f"git clone https://github.com/pjreddie/darknet.git {pj_dir}")
    out, rc = run_and_capture("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
    cap = out.strip().splitlines()[0].replace(".", "") if out.strip() else "75"
    gencode = f"-gencode arch=compute_{cap},code=[sm_{cap},compute_{cap}]"
    print(f"Detected GPU compute capability {cap}")
    makefile_path = os.path.join(pj_dir, "Makefile")
    mk = open(makefile_path).read()
    mk = re.sub(r"^GPU=0", "GPU=1", mk, flags=re.MULTILINE)
    mk = re.sub(r"^CUDNN=1", "CUDNN=0", mk, flags=re.MULTILINE)
    mk = re.sub(r"^OPENCV=1", "OPENCV=0", mk, flags=re.MULTILINE)
    mk = patch_arch_line(mk, gencode)
    with open(makefile_path, "w") as fh: fh.write(mk)
    run_and_capture("make clean", cwd=pj_dir)
    out, rc = run_and_capture("make -j$(nproc)", cwd=pj_dir)
    if rc != 0 or not os.path.isfile(existing):
        print("pjreddie/darknet failed to compile -- see error above.")
        return None
    return existing

_darknet_dataset_cache = {}
def darknet_prepare_dataset():
    """Shared flat image+label layout for v1/v2/v4. Built once, reused."""
    if "dn_data" in _darknet_dataset_cache:
        return _darknet_dataset_cache["dn_data"]
    dn_data = os.path.join(DARKNET_WORK, "darknet_data")
    if os.path.isfile(os.path.join(dn_data, "train.txt")):
        _darknet_dataset_cache["dn_data"] = dn_data
        return dn_data
    os.makedirs(os.path.join(dn_data, "train"), exist_ok=True)
    os.makedirs(os.path.join(dn_data, "val"), exist_ok=True)

    def flatten(split):
        img_src = os.path.join(NEU_YOLO, "images", split)
        lbl_src = os.path.join(NEU_YOLO, "labels", split)
        dst = os.path.join(dn_data, split)
        paths = []
        for img_file in list_files(img_src, (".jpg",".jpeg",".png",".bmp")):
            shutil.copy2(os.path.join(img_src, img_file), os.path.join(dst, img_file))
            shutil.copy2(os.path.join(lbl_src, stem(img_file)+".txt"), os.path.join(dst, stem(img_file)+".txt"))
            paths.append(os.path.join(dst, img_file))
        return paths

    train_paths, val_paths = flatten("train"), flatten("val")
    with open(os.path.join(dn_data, "train.txt"), "w") as fh: fh.write("\n".join(train_paths))
    with open(os.path.join(dn_data, "val.txt"), "w") as fh: fh.write("\n".join(val_paths))
    with open(os.path.join(dn_data, "obj.names"), "w") as fh: fh.write("\n".join(CLASS_NAMES))
    _darknet_dataset_cache["dn_data"] = dn_data
    return dn_data

def darknet_max_batches(n_train, batch, epochs, n_classes):
    return max(epochs * max(1, n_train // batch), 2000 * n_classes)

def patch_net_block(cfg_text, batch, subdivisions, width, height, max_batches):
    lines = cfg_text.splitlines()
    steps = f"{int(max_batches*0.8)},{int(max_batches*0.9)}"
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "[net]":
            out.append(line); i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                l = lines[i]
                if l.strip().startswith("batch="): l = f"batch={batch}"
                elif l.strip().startswith("subdivisions="): l = f"subdivisions={subdivisions}"
                elif l.strip().startswith("width="): l = f"width={width}"
                elif l.strip().startswith("height="): l = f"height={height}"
                elif l.strip().startswith("max_batches"): l = f"max_batches={max_batches}"
                elif l.strip().startswith("steps"): l = f"steps={steps}"
                out.append(l); i += 1
            continue
        out.append(line); i += 1
    return "\n".join(out)

def patch_cfg_yolo_or_region(cfg_text, num_classes, batch, width, height, max_batches,
                              layer_type, subdivisions=16):
    """For v2 ([region]) and v4 ([yolo]). filters on the preceding [convolutional]
    = (classes+5) * num_anchors -- num_anchors comes from 'mask' length for [yolo]
    (v4 uses a subset of anchors per scale) or straight from 'num' for [region]
    (v2 has no mask concept, single scale, all anchors used)."""
    text = patch_net_block(cfg_text, batch, subdivisions, width, height, max_batches)
    tag = layer_type.strip("[]")
    pattern = re.compile(r"(\[" + tag + r"\][^\[]*?classes\s*=\s*)\d+", re.DOTALL)
    text = pattern.sub(lambda m: m.group(1) + str(num_classes), text)
    blocks = re.split(r"(?=\[)", text)
    for idx, block in enumerate(blocks):
        if block.split("]",1)[0]+"]" == f"[{tag}]":
            for j in range(idx-1, -1, -1):
                if blocks[j].strip().startswith("[convolutional]"):
                    if tag == "yolo":
                        mask_match = re.search(r"mask\s*=\s*([\d,]+)", block)
                        n_anchors = len(mask_match.group(1).split(",")) if mask_match else 3
                    else:  # region: use num= directly, no mask
                        num_match = re.search(r"^\s*num\s*=\s*(\d+)", block, re.MULTILINE)
                        n_anchors = int(num_match.group(1)) if num_match else 5
                    new_filters = (num_classes + 5) * n_anchors
                    blocks[j] = re.sub(r"filters\s*=\s*\d+", f"filters={new_filters}", blocks[j])
                    break
    return "".join(blocks)

def patch_cfg_v1_detection(cfg_text, num_classes, batch, width, height, max_batches, subdivisions=16):
    """YOLOv1 uses [connected]+output feeding a [detection] layer, not
    [convolutional]+filters. output = side*side*(classes + num*5)."""
    text = patch_net_block(cfg_text, batch, subdivisions, width, height, max_batches)
    blocks = re.split(r"(?=\[)", text)
    for idx, block in enumerate(blocks):
        if block.strip().startswith("[detection]"):
            side_m = re.search(r"side\s*=\s*(\d+)", block)
            num_m = re.search(r"^\s*num\s*=\s*(\d+)", block, re.MULTILINE)
            side = int(side_m.group(1)) if side_m else 7
            num = int(num_m.group(1)) if num_m else 2
            blocks[idx] = re.sub(r"classes\s*=\s*\d+", f"classes={num_classes}", block)
            new_output = side * side * (num_classes + num * 5)
            for j in range(idx-1, -1, -1):
                if blocks[j].strip().startswith("[connected]"):
                    blocks[j] = re.sub(r"output\s*=\s*\d+", f"output={new_output}", blocks[j])
                    break
            break
    return "".join(blocks)

def darknet_parse_training_log(log_text):
    """Darknet prints per-iteration loss and, every 100 iterations, an mAP line
    (with -map). This is the closest thing to a training curve Darknet offers --
    not per-epoch precision/recall the way the other frameworks give."""
    iters, losses = [], []
    for m in re.finditer(r"^\s*(\d+):\s*[\d.]+,\s*avg loss=([\d.]+),", log_text, re.MULTILINE):
        iters.append(int(m.group(1))); losses.append(float(m.group(2)))
    # fallback pattern seen in some Darknet builds: "N: loss=X, avg loss=Y, ..."
    if not iters:
        for m in re.finditer(r"^\s*(\d+):\s*loss=[\d.]+,\s*avg loss=([\d.]+),", log_text, re.MULTILINE):
            iters.append(int(m.group(1))); losses.append(float(m.group(2)))
    map_iters, map_vals = [], []
    for m in re.finditer(r"\(next mAP calculation at (\d+) iterations\).*?mAP@[\d.]+\s*=\s*([\d.]+)\s*%", log_text, re.DOTALL):
        pass  # format varies too much across builds to trust this reliably
    for m in re.finditer(r"mean_average_precision.*?=\s*([\d.]+)", log_text, re.IGNORECASE):
        map_vals.append(float(m.group(1)))
    return iters, losses, map_vals

def darknet_train(version_tag, model_name, cfg_filename, layer_type, patch_fn,
                   pretrained_url, pretrained_name, darknet_bin_getter):
    """Shared training routine for v1/v2/v4, parameterized by which cfg/layer/
    patch function/pretrained weights/binary-builder to use."""
    darknet_bin = darknet_bin_getter()
    if darknet_bin is None:
        return None, None, None, None

    dn_data = darknet_prepare_dataset()
    backup_dir = os.path.join(MODEL_DIR[model_name]["weights"], "backup")
    if os.path.isdir(backup_dir): shutil.rmtree(backup_dir)
    os.makedirs(backup_dir, exist_ok=True)

    obj_data_path = os.path.join(DARKNET_WORK, f"obj_{version_tag}.data")
    with open(obj_data_path, "w") as fh:
        fh.write(f"""classes = {NUMBER_OF_CLASSES}
train = {os.path.join(dn_data, "train.txt")}
valid = {os.path.join(dn_data, "val.txt")}
names = {os.path.join(dn_data, "obj.names")}
backup = {backup_dir}
""")

    cfg_dir = os.path.join(DARKNET_WORK, f"cfg_{version_tag}")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_raw = os.path.join(cfg_dir, cfg_filename)
    run_and_capture(f"wget -q -O {cfg_raw} https://raw.githubusercontent.com/hank-ai/darknet/master/cfg/{cfg_filename}")
    if not os.path.isfile(cfg_raw) or os.path.getsize(cfg_raw) == 0:
        # v1 fallback source: pjreddie's repo still has the legacy cfg
        run_and_capture(f"wget -q -O {cfg_raw} https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/{cfg_filename}")
    if not os.path.isfile(cfg_raw) or os.path.getsize(cfg_raw) == 0:
        print(f"Could not download {cfg_filename} from either repo.")
        return None, None, None, None

    n_train = len(list_files(os.path.join(NEU_YOLO,"images","train"), (".jpg",".jpeg",".png",".bmp")))
    max_batches = darknet_max_batches(n_train, BATCH_SIZE, EPOCHS, NUMBER_OF_CLASSES)
    patched = patch_fn(open(cfg_raw).read(), NUMBER_OF_CLASSES, batch=BATCH_SIZE,
                        width=IMAGE_SIZE, height=IMAGE_SIZE, max_batches=max_batches)
    cfg_path = os.path.join(cfg_dir, f"{version_tag}_custom.cfg")
    with open(cfg_path, "w") as fh: fh.write(patched)
    print(f"Patched cfg: max_batches={max_batches} (~{EPOCHS} epochs equivalent)")

    weights_path = os.path.join(MODEL_DIR[model_name]["pretrained"], pretrained_name)
    if not os.path.isfile(weights_path):
        out, rc = run_and_capture(f"wget -q -O {weights_path} {pretrained_url}")
        if rc != 0 or os.path.getsize(weights_path) == 0:
            print("Pretrained backbone weights unavailable -- training from random init.")
            weights_path = ""

    if weights_path:
        cmd = f"{darknet_bin} detector train {obj_data_path} {cfg_path} {weights_path} -dont_show -map"
    else:
        cmd = f"{darknet_bin} detector train {obj_data_path} {cfg_path} -dont_show -map"
    t0 = time.time()
    log_text, rc = run_and_capture(cmd, cwd=cfg_dir)
    training_time_s = time.time() - t0
    return darknet_bin, obj_data_path, cfg_path, (backup_dir, training_time_s, log_text)

def darknet_evaluate(darknet_bin, obj_data_path, cfg_path, backup_dir, training_time_s):
    best_candidates = [f for f in os.listdir(backup_dir) if f.endswith("_best.weights")] if os.path.isdir(backup_dir) else []
    if not best_candidates:
        return None
    best_weights = os.path.join(backup_dir, sorted(best_candidates, key=lambda f: os.path.getmtime(os.path.join(backup_dir,f)))[-1])

    out50, rc = run_and_capture(f"{darknet_bin} detector map {obj_data_path} {cfg_path} {best_weights}", cwd=os.path.dirname(cfg_path))
    mAP50_raw = parse_first_float(r"mean average precision.*?=\s*([\d.]+)\s*%?", out50)
    mAP50 = (mAP50_raw/100.0) if mAP50_raw is not None else None
    precision = parse_first_float(r"precision\s*=\s*([\d.]+)", out50)
    recall = parse_first_float(r"recall\s*=\s*([\d.]+)", out50)
    f1 = parse_first_float(r"f1-score\s*=\s*([\d.]+)", out50)
    tp = parse_first_float(r"TP\s*=\s*(\d+)", out50)
    fp = parse_first_float(r"FP\s*=\s*(\d+)", out50)

    per_class = {}
    for m in re.finditer(r"class_id\s*=\s*(\d+),\s*name\s*=\s*([\w_]+),\s*ap\s*=\s*([\d.]+)\s*%", out50, re.IGNORECASE):
        per_class[m.group(2)] = float(m.group(3))/100.0

    ious = [0.5+0.05*i for i in range(10)]
    sweep_vals, supported = [], True
    for iou in ious:
        out_i, rc_i = run_and_capture(f"{darknet_bin} detector map {obj_data_path} {cfg_path} {best_weights} -iou_thresh {iou:.2f}", cwd=os.path.dirname(cfg_path))
        if rc_i != 0 or "unknown argument" in out_i.lower():
            supported = False; break
        v = parse_first_float(r"mean average precision.*?=\s*([\d.]+)\s*%?", out_i)
        if v is not None: sweep_vals.append(v)
    mAP50_95 = (sum(sweep_vals)/len(sweep_vals)/100.0) if (supported and sweep_vals) else None

    model_size_mb = os.path.getsize(best_weights)/(1024*1024)
    n_val = len(list_files(os.path.join(NEU_YOLO,"images","val"), (".jpg",".jpeg",".png",".bmp")))

    overall = {
        "Framework": "Darknet", "Image_Size": IMAGE_SIZE, "Batch_Size": BATCH_SIZE, "Epochs": EPOCHS,
        "Precision": precision, "Recall": recall, "F1": f1, "mAP50": mAP50, "mAP50_95": mAP50_95,
        "TP": tp, "FP": fp, "Num_Val_Images": n_val,
        "FPS": None, "Inference_Time_ms": None, "Parameters": None,
        "Model_Size_MB": model_size_mb, "Training_Time_s": training_time_s,
        "AP_patches": per_class.get("patches"),
        "AP_pitted_surface": per_class.get("pitted_surface"), "AP_scratches": per_class.get("scratches"),
        "Status": "SUCCESS",
        "Notes": ("mAP50_95 approximated by averaging map at 10 IoU thresholds." if mAP50_95 is not None
                  else "mAP50_95 not available: this build's map tool did not accept -iou_thresh.")
                 + " Parameters/FPS/inference time not available from Darknet's .weights format.",
    }
    per_class_rows = [{"Model": None, "Class": c, "Precision": precision, "Recall": recall,
                        "F1": f1, "AP50": per_class.get(c)} for c in CLASS_NAMES]
    return overall, per_class_rows, best_weights

def darknet_inference(darknet_bin, obj_data_path, cfg_path, weights_path, image_path, out_dir, model_name):
    run_and_capture(f"{darknet_bin} detector test {obj_data_path} {cfg_path} {weights_path} {image_path} -dont_show", cwd=os.path.dirname(cfg_path))
    pred_src = os.path.join(os.path.dirname(cfg_path), "predictions.jpg")
    detections = []
    if os.path.isfile(pred_src):
        out_path = os.path.join(out_dir, f"{model_name}_{os.path.basename(image_path)}")
        shutil.copy2(pred_src, out_path)
        plt.figure(figsize=(8,8)); plt.imshow(Image.open(out_path)); plt.axis("off")
        plt.title(f"{model_name} inference"); plt.show()
        print(f"Saved: {out_path}")
        return out_path, detections
    print("No predictions.jpg produced -- inference may have failed.")
    return None, detections

print("Darknet-family functions loaded: build_modern_darknet, build_pjreddie_darknet, "
      "darknet_prepare_dataset, patch_cfg_yolo_or_region, patch_cfg_v1_detection, "
      "darknet_train, darknet_evaluate, darknet_inference, darknet_parse_training_log")


# ======================================================================
# CODE CELL 12
# ======================================================================
# ============================= YOLOv6 FAMILY =============================

def yolov6_setup(model_name):
    repo_dir = MODEL_DIR[model_name]["repo"]
    if not os.listdir(repo_dir):
        run_and_capture(f"git clone https://github.com/meituan/YOLOv6.git {repo_dir}")
        run_and_capture("pip install -q -r requirements.txt", cwd=repo_dir)
    yaml_path = os.path.join(repo_dir, "data", "neu_det.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"""train: {os.path.join(NEU_YOLO, "images", "train")}
val: {os.path.join(NEU_YOLO, "images", "val")}
test: {os.path.join(NEU_YOLO, "images", "val")}
nc: {NUMBER_OF_CLASSES}
names: {CLASS_NAMES}
""")
    return repo_dir, yaml_path

def yolov6_train(model_name, repo_dir, yaml_path):
    pretrained_pt = os.path.join(MODEL_DIR[model_name]["pretrained"], "yolov6s.pt")
    if not os.path.isfile(pretrained_pt):
        run_and_capture(f"wget -q -O {pretrained_pt} https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt")
    run_dir = MODEL_DIR[model_name]["weights"]
    cmd = (f"python tools/train.py --data-path {yaml_path} --conf-file configs/yolov6s.py "
           f"--img-size {IMAGE_SIZE} --epochs {EPOCHS} --batch-size {BATCH_SIZE} "
           f"--device 0 --output-dir {run_dir} --name run")
    t0 = time.time()
    out, rc = run_and_capture(cmd, cwd=repo_dir)
    training_time_s = time.time() - t0
    return run_dir, training_time_s, rc

def yolov6_evaluate(repo_dir, yaml_path, run_dir, training_time_s):
    weights = os.path.join(run_dir, "run", "weights", "best_ckpt.pt")
    if not os.path.isfile(weights):
        return None, None
    cmd = f"python tools/eval.py --data {yaml_path} --weights {weights} --img-size {IMAGE_SIZE} --batch-size {BATCH_SIZE} --device 0"
    out, rc = run_and_capture(cmd, cwd=repo_dir)

    mAP50 = parse_first_float(r"ap50[^\d]*([\d.]+)", out)
    mAP50_95 = parse_first_float(r"ap50_?95[^\d]*([\d.]+)", out) or parse_first_float(r"ap50:95[^\d]*([\d.]+)", out)
    precision = parse_first_float(r"precision[^\d]*([\d.]+)", out)
    recall = parse_first_float(r"recall[^\d]*([\d.]+)", out)
    f1 = (2*precision*recall/(precision+recall)) if (precision and recall and (precision+recall)>0) else None
    per_class = {cls: parse_first_float(rf"{cls}[^\d]*([\d.]+)", out) for cls in CLASS_NAMES}
    model_size_mb = os.path.getsize(weights)/(1024*1024)

    overall = {
        "Framework": "YOLOv6 (Meituan)", "Pretrained_Weights": "yolov6s.pt",
        "Image_Size": IMAGE_SIZE, "Batch_Size": BATCH_SIZE, "Epochs": EPOCHS,
        "Precision": precision, "Recall": recall, "F1": f1, "mAP50": mAP50, "mAP50_95": mAP50_95,
        "Model_Size_MB": model_size_mb, "Training_Time_s": training_time_s,
        "AP_patches": per_class.get("patches"),
        "AP_pitted_surface": per_class.get("pitted_surface"), "AP_scratches": per_class.get("scratches"),
        "Status": "SUCCESS",
        "Notes": "Metrics regex-parsed from YOLOv6 console output. Parameters/FPS/inference "
                 "time not extracted by this cell.",
    }
    return overall, weights

def yolov6_curves(model_name, run_dir):
    """YOLOv6 typically logs per-epoch metrics to a results file near its run
    directory; format/filename can vary across releases, so this searches for it
    rather than assuming a fixed path, and reports plainly if not found."""
    candidates = []
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f in ("results.csv", "results.txt"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        print("No results.csv/results.txt found under this YOLOv6 run -- per-epoch "
              "training curves are not available for this release of the repo.")
        return
    path = candidates[0]
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        graphs_dir = MODEL_DIR[model_name]["graphs"]
        for col in df.columns:
            if col.lower() in ("epoch",): continue
            plt.figure(figsize=(7,4))
            plt.plot(df[df.columns[0]], df[col], marker="o", markersize=3)
            plt.xlabel(df.columns[0]); plt.ylabel(col); plt.title(f"{model_name}: {col} vs epoch")
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, f"{col}.png".replace('/','_')))
            plt.show()
    except Exception as e:
        print(f"Found {path} but could not parse it as a table ({e}) -- YOLOv6's log "
              f"format may differ from what this cell expects for this repo version.")

def yolov6_inference(repo_dir, weights, image_path, out_dir, model_name):
    sample_src = "/content/_yolov6_infer_input"
    os.makedirs(sample_src, exist_ok=True)
    shutil.copy2(image_path, os.path.join(sample_src, os.path.basename(image_path)))
    cmd = f"python tools/infer.py --weights {weights} --source {sample_src} --img-size {IMAGE_SIZE} --save-dir {out_dir} --device 0"
    run_and_capture(cmd, cwd=repo_dir)
    out_path = os.path.join(out_dir, os.path.basename(image_path))
    if os.path.isfile(out_path):
        plt.figure(figsize=(8,8)); plt.imshow(Image.open(out_path)); plt.axis("off")
        plt.title(f"{model_name} inference"); plt.show()
        print(f"Saved: {out_path}")
        return out_path, []
    print("Inference did not produce an output image at the expected path.")
    return None, []

# ============================= YOLOv7 FAMILY =============================

def yolov7_setup(model_name):
    repo_dir = MODEL_DIR[model_name]["repo"]
    if not os.listdir(repo_dir):
        run_and_capture(f"git clone https://github.com/WongKinYiu/yolov7.git {repo_dir}")
        out, rc = run_and_capture("pip install -q -r requirements.txt", cwd=repo_dir)
        if rc != 0:
            print("pip install for yolov7 reported problems -- known torch/numpy pin friction.")
    yaml_path = os.path.join(repo_dir, "data", "neu_det.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(f"""train: {os.path.join(NEU_YOLO, "images", "train")}
val: {os.path.join(NEU_YOLO, "images", "val")}
nc: {NUMBER_OF_CLASSES}
names: {CLASS_NAMES}
""")
    return repo_dir, yaml_path

def yolov7_train(model_name, repo_dir, yaml_path):
    pt_path = os.path.join(MODEL_DIR[model_name]["pretrained"], "yolov7.pt")
    if not os.path.isfile(pt_path):
        run_and_capture(f"wget -q -O {pt_path} https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
    run_dir = MODEL_DIR[model_name]["weights"]
    cmd = (f"python train.py --workers 2 --device 0 --batch-size {BATCH_SIZE} "
           f"--data {yaml_path} --img {IMAGE_SIZE} {IMAGE_SIZE} --cfg cfg/training/yolov7.yaml "
           f"--weights {pt_path} --epochs {EPOCHS} --project {run_dir} --name run --exist-ok")
    t0 = time.time()
    out, rc = run_and_capture(cmd, cwd=repo_dir)
    training_time_s = time.time() - t0
    return run_dir, training_time_s, rc

def yolov7_evaluate(repo_dir, yaml_path, run_dir, training_time_s):
    weights = os.path.join(run_dir, "run", "weights", "best.pt")
    if not os.path.isfile(weights):
        return None, None
    cmd = f"python test.py --data {yaml_path} --img {IMAGE_SIZE} --batch {BATCH_SIZE} --weights {weights} --task val"
    out, rc = run_and_capture(cmd, cwd=repo_dir)

    m = re.search(r"\ball\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
    precision = recall = mAP50 = mAP50_95 = None
    if m:
        precision, recall, mAP50, mAP50_95 = (float(g) for g in m.groups())
    f1 = (2*precision*recall/(precision+recall)) if (precision and recall and (precision+recall)>0) else None

    per_class = {}
    for cls in CLASS_NAMES:
        mm = re.search(rf"\b{cls}\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
        per_class[cls] = float(mm.group(3)) if mm else None
    model_size_mb = os.path.getsize(weights)/(1024*1024)

    overall = {
        "Framework": "YOLOv7 (WongKinYiu)", "Pretrained_Weights": "yolov7.pt",
        "Image_Size": IMAGE_SIZE, "Batch_Size": BATCH_SIZE, "Epochs": EPOCHS,
        "Precision": precision, "Recall": recall, "F1": f1, "mAP50": mAP50, "mAP50_95": mAP50_95,
        "Model_Size_MB": model_size_mb, "Training_Time_s": training_time_s,
        "AP_patches": per_class.get("patches"),
        "AP_pitted_surface": per_class.get("pitted_surface"), "AP_scratches": per_class.get("scratches"),
        "Status": "SUCCESS",
        "Notes": "Metrics regex-parsed from YOLOv7 console summary table. Parameters/FPS/"
                 "inference time not extracted by this cell.",
    }
    return overall, weights

def yolov7_curves(model_name, repo_dir):
    """YOLOv7's train.py (inherited from the YOLOv5 codebase) typically saves
    results.csv/results.txt in its run directory."""
    run_dir = os.path.join(repo_dir, "runs", "train")
    candidates = []
    if os.path.isdir(run_dir):
        for root, dirs, files in os.walk(run_dir):
            for f in files:
                if f in ("results.csv", "results.txt"):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        print("No results.csv/results.txt found under this YOLOv7 run -- per-epoch "
              "training curves are not available.")
        return
    path = candidates[0]
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        df.columns = [c.strip() for c in df.columns]
        graphs_dir = MODEL_DIR[model_name]["graphs"]
        for col in df.columns[1:]:
            plt.figure(figsize=(7,4))
            plt.plot(df[df.columns[0]], df[col], marker="o", markersize=3)
            plt.xlabel(df.columns[0]); plt.ylabel(col); plt.title(f"{model_name}: {col} vs epoch")
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, f"{col}.png".replace('/','_')))
            plt.show()
    except Exception as e:
        print(f"Found {path} but could not parse it ({e}).")

def yolov7_confusion_pr(model_name, repo_dir):
    eval_run_dir = os.path.join(repo_dir, "runs", "test")
    plot_names = ["confusion_matrix.png", "PR_curve.png", "P_curve.png", "R_curve.png", "F1_curve.png"]
    found = {}
    if os.path.isdir(eval_run_dir):
        for root, dirs, files in os.walk(eval_run_dir):
            for p in plot_names:
                if p in files and p not in found:
                    found[p] = os.path.join(root, p)
    graphs_dir = MODEL_DIR[model_name]["graphs"]
    for name, src in found.items():
        shutil.copy2(src, os.path.join(graphs_dir, name))
    print(f"Copied {list(found.keys())}" if found else "No confusion matrix / PR curve files found.")
    missing = [p for p in plot_names if p not in found]
    if missing: print(f"Not found: {missing}")

def yolov7_inference(repo_dir, weights, image_path, out_dir, model_name):
    sample_src = "/content/_yolov7_infer_input"
    os.makedirs(sample_src, exist_ok=True)
    shutil.copy2(image_path, os.path.join(sample_src, os.path.basename(image_path)))
    cmd = (f"python detect.py --weights {weights} --source {sample_src} --img-size {IMAGE_SIZE} "
           f"--project {out_dir} --name samples --exist-ok")
    run_and_capture(cmd, cwd=repo_dir)
    out_path = os.path.join(out_dir, "samples", os.path.basename(image_path))
    if os.path.isfile(out_path):
        plt.figure(figsize=(8,8)); plt.imshow(Image.open(out_path)); plt.axis("off")
        plt.title(f"{model_name} inference"); plt.show()
        print(f"Saved: {out_path}")
        return out_path, []
    print("Inference did not produce an output image at the expected path.")
    return None, []

print("YOLOv6-family functions loaded: yolov6_setup, yolov6_train, yolov6_evaluate, yolov6_curves, yolov6_inference")
print("YOLOv7-family functions loaded: yolov7_setup, yolov7_train, yolov7_evaluate, yolov7_curves, yolov7_confusion_pr, yolov7_inference")


# ======================================================================
# CODE CELL 13
# ======================================================================
from ultralytics import YOLO
project_dir_YOLOv11 = os.path.join(MODEL_DIR["YOLOv11"]["weights"])
print("YOLOv11 ready. Pretrained weights: yolo11n.pt")
print("Output directory:", project_dir_YOLOv11)


# ======================================================================
# CODE CELL 14
# ======================================================================
model_YOLOv11, YOLOv11_training_time_s = ultralytics_train("YOLOv11", "yolo11n.pt", project_dir_YOLOv11)
YOLOv11_ready = True
try:
    _args = model_YOLOv11.trainer.args
    print(f"Optimizer: {getattr(_args, 'optimizer', 'not available')}")
    print(f"Learning rate (lr0): {getattr(_args, 'lr0', 'not available')}")
except Exception:
    print("Optimizer/learning rate: not available from trainer args for this version.")
print(f"\nTraining time: {YOLOv11_training_time_s:.1f} s")


# ======================================================================
# CODE CELL 15
# ======================================================================
YOLOv11_overall, YOLOv11_per_class, YOLOv11_metrics = ultralytics_evaluate(
    "YOLOv11", model_YOLOv11, "yolo11n.pt", YOLOv11_training_time_s)
YOLOv11_overall.update({"Architecture": "Anchor-free, decoupled head; C3k2 backbone blocks plus C2PSA partial self-attention", "Backbone": "C3k2 + C2PSA"})
for row in YOLOv11_per_class: row["Model"] = "YOLOv11"
save_overall_result("YOLOv11", YOLOv11_overall)
save_per_class_result("YOLOv11", YOLOv11_per_class)
print(YOLOv11_overall)


# ======================================================================
# CODE CELL 16
# ======================================================================
ultralytics_curves("YOLOv11", project_dir_YOLOv11)
ultralytics_confusion_pr("YOLOv11", YOLOv11_metrics)


# ======================================================================
# CODE CELL 17
# ======================================================================
from google.colab import files
print("Upload an image to test with YOLOv11:")
uploaded = files.upload()
for fname in uploaded.keys():
    img_path = os.path.join("/content", fname)
    ultralytics_inference(model_YOLOv11, img_path, MODEL_DIR["YOLOv11"]["inference_results"], "YOLOv11")

