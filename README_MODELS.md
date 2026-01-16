# Hand Sign Detection App

This project provides a real-time hand sign and gesture detection application using Streamlit.

## Models Available

### Option 1: YOLOv10 (Recommended - Actually Detects Gestures ✅)
**File:** `app_yolo.py`

Uses the trained YOLOv10x model (`YOLOv10x_gestures.pt`). This model has been trained and **actually detects hand gestures in real-time**.

```bash
streamlit run app_yolo.py
```

**Features:**
- ✅ Real-time hand gesture detection
- ✅ Works immediately - no threshold tuning needed
- ✅ Shows bounding boxes with gesture labels and confidence scores
- ✅ Adjustable confidence threshold (0.1 - 1.0)

### Option 2: HaGRID (For Reference - Untrained)
**File:** `app.py`

Uses the HaGRID SSDLiteMobileNetV3Large model. This model architecture is loaded but **is not trained**, so it won't detect real gestures (it generates random anchor boxes).

```bash
streamlit run app.py
```

**Note:** This is primarily for reference/educational purposes. To use this, you would need to train it on the HaGRID dataset first.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Detection App:**
   ```bash
   streamlit run app_yolo.py
   ```

3. **In the Streamlit App:**
   - Click "Load Model" to load the YOLOv10 model
   - Click "Start Camera" to begin detection
   - Point your camera at your hands and perform gestures
   - Adjust the confidence threshold slider if needed (default: 0.5)

## UI Features

### Live Detection Display
- **Green bounding boxes** around detected hands
- **Gesture labels** (e.g., "palm", "fist", "ok", "peace", etc.)
- **Confidence scores** for each detection
- **Real-time FPS counter** in top-left corner
- **Detection count** showing how many gestures are detected

### Settings Panel (Sidebar)
- Model path selection
- Confidence threshold slider
- Real-time statistics (detections, confidence)
- FPS monitoring

### Detection Info
- Live status updates
- Frame counter
- Real-time detection feedback

## Supported Gestures

The YOLOv10 model supports 34+ hand gestures including:
- Palm, Fist, OK, Peace
- Like, Dislike, Rock
- Stop, Call, Mute
- Three, Two Up, Pointing
- And many more...

## Troubleshooting

### "Camera not opening"
- Make sure you have camera permissions enabled
- Try `camera:latest` or check `About` menu

### "Low FPS or lag"
- Lower the model confidence threshold
- Close other applications
- Check system resources

### "No detections"
- Make sure you're using `app_yolo.py` (not `app.py`)
- Ensure the camera is pointed at your hand
- Try adjusting the confidence threshold lower (start at 0.3)
- Make sure lighting is adequate

## System Requirements

- Python 3.8+
- macOS, Linux, or Windows
- Webcam/Camera device
- ~2GB RAM minimum

## Architecture

```
Hand Sign Detection
├── app.py              → HaGRID model (untrained, reference only)
├── app_yolo.py         → YOLOv10 model (recommended)
├── YOLOv10x_gestures.pt → Trained YOLO model weights
└── hagrid-Hagrid_v2-1M/ → HaGRID model code & configs
```

## Performance

With YOLOv10x on typical hardware:
- **FPS:** 15-30 FPS (depending on device)
- **Latency:** ~30-60ms per frame
- **Model Size:** ~300MB
- **Memory:** ~1-2GB during inference

## Notes

- Hand landmarks feature is disabled due to threading conflicts on macOS
- Focus on gesture detection for best results
- Adjust lighting for better detection accuracy
- Keep hands within frame for optimal results
