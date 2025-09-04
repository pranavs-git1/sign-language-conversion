import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models

# Scikit
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# MediaPipe Hands
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception:
    print("\n[ERROR] MediaPipe not found. Install with: pip install mediapipe\n")
    raise


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = DATA_DIR / "isl_data.csv"
MODEL_PATH = MODEL_DIR / "isl_tf_model.keras"
LABELS_PATH = MODEL_DIR / "labels.json"



ISL_LABELS = [
    "hello", "thankyou", "please", "sorry", "love",
    "yes", "no", "ok", "stop",
    "eat", "drink", "water", "toilet", "help",
    "who", "what", "where", "when", "why", "how",
    "i", "you", "we", "they",
    "book", "phone", "money",
]


def landmarks_to_features(hand_landmarks, image_w: int, image_h: int) -> np.ndarray:

    pts = []
    for lm in hand_landmarks.landmark:
        x_px = lm.x * image_w
        y_px = lm.y * image_h
        z_rel = lm.z
        pts.append([x_px, y_px, z_rel])
    pts = np.array(pts, dtype=np.float32)  # (21,3)


    wrist = pts[0].copy()
    pts[:, :2] -= wrist[:2]
    pts[:, 2] -= wrist[2]


    xy = pts[:, :2]
    diffs = xy[:, None, :] - xy[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    max_dist = float(np.max(dists))
    if max_dist < 1e-6:
        scale = 1.0
    else:
        scale = max_dist
    pts /= scale
    return pts  # (21,3)


def extract_features_from_frame(frame_bgr, hands_detector):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_bgr.shape[:2]
    result = hands_detector.process(img_rgb)
    feats = []
    bboxes = []
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            vec = landmarks_to_features(hand, w, h)
            feats.append(vec)
            pts = np.array([[lm.x*w, lm.y*h] for lm in hand.landmark])
            x1,y1 = pts.min(axis=0)
            x2,y2 = pts.max(axis=0)
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return feats, bboxes


def ensure_csv_header():
    if not CSV_PATH.exists():
        cols = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]
        pd.DataFrame(columns=cols).to_csv(CSV_PATH, index=False)


def append_sample(pts_21x3: np.ndarray, label: str):
    x = pts_21x3[:,0].tolist()
    y = pts_21x3[:,1].tolist()
    z = pts_21x3[:,2].tolist()
    row = x + y + z + [label]
    df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, mode='a', header=False, index=False)


def load_dataset():
    if not CSV_PATH.exists():
        print(f"[ERROR] No dataset at {CSV_PATH}. Run collect first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        print("[ERROR] Dataset is empty.")
        sys.exit(1)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    # reshape to (N,21,3)
    X = np.stack([
        np.stack([X[:,0:21], X[:,21:42], X[:,42:63]], axis=-1)
    ], axis=1).squeeze(1)
    labels = df["label"].astype(str).values


    if LABELS_PATH.exists():
        with open(LABELS_PATH, 'r') as f:
            labels_list = json.load(f)
    else:
        labels_list = sorted(np.unique(labels).tolist())
        with open(LABELS_PATH, 'w') as f:
            json.dump(labels_list, f, indent=2)
    label_to_idx = {l:i for i,l in enumerate(labels_list)}

    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
    return X, y, labels_list


def build_model(num_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(21,3))
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(96, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def run_collect(label: str, device: int, num_frames: int, max_hands: int, interval: float):
    if not label:
        print("[ERROR] --label required. Choose one from ISL_LABELS or add your own.")
        sys.exit(1)
    ensure_csv_header()

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=max_hands,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        saved = 0
        last = 0.0
        print(f"[INFO] Collecting label='{label}'. Press 'q' to stop. Target frames={num_frames or '∞'}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed.")
                break
            feats, bboxes = extract_features_from_frame(frame, hands)

            # Draw bboxes
            for (x1,y1,x2,y2) in bboxes:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} | saved:{saved}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Collect (ISL)", frame)

            now = time.time()
            if feats and (now - last) >= interval:
                # Use most prominent hand if multiple
                if len(feats) > 1 and bboxes:
                    areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in bboxes]
                    idx = int(np.argmax(areas))
                    vec = feats[idx]
                else:
                    vec = feats[0]
                append_sample(vec, label)
                saved += 1
                last = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if num_frames > 0 and saved >= num_frames: break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {saved} samples to {CSV_PATH}")


def run_train(test_size: float, epochs: int, batch: int, val_split_cli: float, augment: bool):
    X, y, labels_list = load_dataset()

    # Split train/holdout first (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Optional tiny augmentation: jitter + slight scale
    def augment_batch(arr):
        noise = np.random.normal(0, 0.015, size=arr.shape).astype(np.float32)
        scale = np.random.uniform(0.95, 1.05, size=(arr.shape[0],1,1)).astype(np.float32)
        return arr * scale + noise

    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(len(X_tr), seed=42).batch(batch)
    if augment:
        train_ds = train_ds.map(lambda a,b: (tf.numpy_function(augment_batch, [a], tf.float32), b), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_te, y_te)).batch(batch)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw = {int(i): float(w) for i,w in enumerate(class_weights)}

    model = build_model(num_classes=len(labels_list))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds if val_split_cli <= 0 else None,
        epochs=epochs,
        class_weight=cw,
        verbose=1,
        callbacks=callbacks
    )

    # Save labels order
    with open(LABELS_PATH, 'w') as f:
        json.dump(labels_list, f, indent=2)

    # Evaluate on holdout
    te_metrics = model.evaluate(val_ds, verbose=0)
    print(f"[EVAL] Loss={te_metrics[0]:.4f}  Acc={te_metrics[1]:.4f}")
    print(f"[SAVED] Model → {MODEL_PATH}\n[INFO] Labels → {LABELS_PATH}")


def run_predict(device: int, max_hands: int, threshold: float, smooth: int):
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("[ERROR] Model/labels not found. Train first with --mode train.")
        sys.exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    labels_list = json.loads(Path(LABELS_PATH).read_text())

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    from collections import deque, Counter
    buf = deque(maxlen=max(1, smooth))

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=max_hands,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame grab failed.")
                break
            feats, bboxes = extract_features_from_frame(frame, hands)

            display = "No hand"
            if feats:
                # choose most prominent
                if len(feats) > 1 and bboxes:
                    areas = [(x2-x1)*(y2-y1) for (x1,y1,x2,y2) in bboxes]
                    idx = int(np.argmax(areas))
                    f = feats[idx]
                    bbox = bboxes[idx]
                else:
                    f = feats[0]
                    bbox = bboxes[0] if bboxes else None

                logits = model.predict(f[np.newaxis, ...], verbose=0)[0]
                p = float(np.max(logits))
                pred_idx = int(np.argmax(logits))
                pred_label = labels_list[pred_idx]

                if p >= threshold:
                    buf.append(pred_label)
                    c = Counter(buf)
                    final, _ = c.most_common(1)[0]
                    display = f"{final} ({p*100:.1f}%)"
                else:
                    display = "Low confidence"

                if bbox is not None:
                    x1,y1,x2,y2 = bbox
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

            cv2.putText(frame, display, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.imshow("Predict (ISL, TF)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="ISL Sign Language — collect/train/predict (TensorFlow)")
    p.add_argument("--mode", required=True, choices=["collect","train","predict"], help="Run mode")

    # Collect
    p.add_argument("--label", type=str, default="", help="Label to record in collect mode (e.g., hello)")
    p.add_argument("--device", type=int, default=0, help="Webcam index")
    p.add_argument("--num_frames", type=int, default=0, help="Stop after this many frames (0=∞)")
    p.add_argument("--max_hands", type=int, default=1, help="Max hands to track")
    p.add_argument("--interval", type=float, default=0.15, help="Seconds between samples")

    # Train
    p.add_argument("--test_size", type=float, default=0.2, help="Holdout fraction for evaluation")
    p.add_argument("--epochs", type=int, default=35, help="Epochs")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--val_split", type=float, default=0.0, help="(unused; kept for compatibility)")
    p.add_argument("--augment", action='store_true', help="Enable small jitter/scale augmentation")

    # Predict
    p.add_argument("--threshold", type=float, default=0.55, help="Min prob to accept a prediction")
    p.add_argument("--smooth", type=int, default=5, help="Majority-vote window size")

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "collect":
        run_collect(label=args.label, device=args.device, num_frames=args.num_frames,
                    max_hands=args.max_hands, interval=args.interval)
    elif args.mode == "train":
        run_train(test_size=args.test_size, epochs=args.epochs, batch=args.batch,
                  val_split_cli=args.val_split, augment=args.augment)
    elif args.mode == "predict":
        run_predict(device=args.device, max_hands=args.max_hands,
                    threshold=args.threshold, smooth=args.smooth)
    else:
        print("Unknown mode.")


if __name__ == "__main__":
    main()
