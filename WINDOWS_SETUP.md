# Running Hand Sign Detection on Windows

## Prerequisites

- **Python 3.8 or higher** (preferably 3.10 or 3.11)
- **pip** (Python package manager)
- **Webcam/Camera** connected to your computer
- **~2GB RAM** available
- **Administrator access** (for driver installation if needed)

## Step-by-Step Installation

### 1. Install Python

1. Download Python from https://www.python.org/downloads/
2. **Important:** Check the box **"Add Python to PATH"** during installation
3. Click **Install Now**
4. Verify installation by opening Command Prompt and running:
   ```bash
   python --version
   ```

### 2. Clone or Download Project

1. Open Command Prompt (Win + R, type `cmd`, press Enter)
2. Navigate to where you want to save the project:
   ```bash
   cd Desktop
   ```
3. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hand-sign-v3
   ```
   
   OR manually download and extract the ZIP file to a folder.

### 3. Create Virtual Environment (Recommended)

A virtual environment keeps dependencies isolated and prevents conflicts:

```bash
# Navigate to project folder
cd hand-sign-v3

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows Command Prompt:
venv\Scripts\activate
# OR on Windows PowerShell:
venv\Scripts\Activate.ps1
```

You should see `(venv)` appear at the beginning of the command line.

### 4. Install Dependencies

With the virtual environment activated, install required packages:

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

**If installation is slow:** You can add `--no-cache-dir` flag:
```bash
pip install --no-cache-dir -r requirements.txt
```

**Dependencies installed:**
- streamlit (web framework)
- opencv-python (camera & image processing)
- ultralytics (YOLO model)
- torch (PyTorch deep learning)
- numpy, pillow (image handling)
- And others...

### 5. Verify Installation

Check if everything is installed correctly:

```bash
python -c "import streamlit, cv2, ultralytics, torch; print('✓ All dependencies installed!')"
```

## Running the App

### Basic Usage

```bash
# Make sure virtual environment is activated (you should see (venv) in prompt)
streamlit run app_yolo.py
```

### What Happens

1. Streamlit will compile and start the app
2. You'll see output like:
   ```
   Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.
   
   You can now view your app in your browser.
   
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

3. **Automatically opens in your browser** OR you can manually open:
   - **Local:** http://localhost:8501
   - **Network:** http://192.168.x.x:8501 (on other devices on same network)

### Using the App

1. **Load Model**: Click the "📦 Load Model" button
   - First time will download the YOLOv10 model (~300MB)
   - Wait for "✅ Model loaded successfully!" message

2. **Start Camera**: Click "▶️ Start Camera"
   - App will request camera permission (allow it)
   - Video feed starts in the main area

3. **Make Gestures**: Point your hand at the camera
   - You should see green bounding boxes around detected hands
   - Gesture names and confidence scores appear above boxes
   - Adjust the confidence slider if needed (0.1 = more detections, 1.0 = fewer/confident)

4. **Stop**: Click "⏹️ Stop Camera" to stop detection

## Stopping the App

In the Command Prompt where the app is running, press:
```
Ctrl + C
```

The app will stop and you'll return to the command prompt.

## Troubleshooting for Windows

### Problem: "Python not found" or "python: command not found"

**Solution:** Python wasn't added to PATH
- Reinstall Python
- **Make sure to check "Add Python to PATH"** during installation
- Restart Command Prompt after installation

### Problem: "Module not found: streamlit/torch/cv2"

**Solution:** Dependencies not installed or using wrong Python
```bash
# Check Python version
python --version

# Check if virtual environment is activated (should show (venv) in prompt)

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Problem: Camera permission denied / Camera not working

**Solution:**
1. Check Settings → Privacy & Security → Camera
2. Scroll down and verify "Allow apps to access your camera" is ON
3. Make sure Streamlit/Python has camera permission
4. Try restarting the app or your computer
5. Test camera with Windows Camera app first

### Problem: "CUDA out of memory" or very low FPS

**Solution:** Your GPU doesn't have enough memory
- Lower confidence threshold (easier detections)
- Close other applications
- Use CPU instead of GPU (already default on most systems)

### Problem: Port 8501 already in use

**Solution:** Another app is using that port
```bash
# Use a different port
streamlit run app_yolo.py --server.port 8502
```

### Problem: App loads but no video appears

**Solution:**
1. Check camera with Windows Camera app first
2. Verify camera permissions in Settings
3. Check if other apps are using the camera (close them)
4. Restart the app

### Problem: Very slow FPS or app freezes

**Solution:**
1. Check system resources (Task Manager - Ctrl + Shift + Esc)
2. Close other heavy applications
3. Lower model confidence threshold
4. Close browser tabs
5. Restart app

## Advanced Options

### Change Camera Device

If you have multiple cameras:
```bash
# List available cameras (create test_camera.py):
```

Create `test_camera.py`:
```python
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        cap.release()
    else:
        print(f"Camera {i}: Not available")
```

Run it:
```bash
python test_camera.py
```

Then edit `app_yolo.py` line ~112, change:
```python
cap = cv2.VideoCapture(0)  # Change 0 to your camera number
```

### Use Different Model Path

If you have a different YOLO model:
1. Open app in browser
2. In the sidebar, change "Model Path" to your file
3. Click "Load Model"

### Disable GPU (if having issues)

The app uses CPU by default. If you want to explicitly disable GPU:

Edit `app_yolo.py` and add after imports:
```python
import torch
torch.cuda.is_available = lambda: False
```

## Performance Tips for Windows

1. **Close unnecessary programs** before running
2. **Disable animations** in Settings → Ease of Access → Display
3. **Use a dedicated webcam** (better quality, better performance)
4. **Good lighting** improves detection accuracy
5. **Keep hand in center** of frame for best results
6. **Adjust confidence threshold** - lower = more detections but more false positives

## Deactivating Virtual Environment

When done, deactivate the virtual environment:

```bash
deactivate
```

You'll see `(venv)` disappear from the command prompt.

## Next Time You Run

Just run these 2 commands:

```bash
# Navigate to project
cd path/to/hand-sign-v3

# Activate virtual environment
venv\Scripts\activate

# Run app
streamlit run app_yolo.py
```

## Getting Help

If you encounter issues:

1. **Check error message** - it usually tells you what's wrong
2. **Try the Troubleshooting section** above
3. **Make sure:**
   - Virtual environment is activated
   - All dependencies installed (`pip list`)
   - Camera works in Windows Camera app
   - You have internet (for first-time model download)

## Windows-Specific Notes

- **First run will be slower** - downloading YOLOv10 model (~300MB)
- **Path separators** - Windows uses `\` instead of `/` (but Python handles both)
- **Antivirus** - May temporarily block file downloads. Add Python to exclusions if needed
- **PowerShell vs CMD** - Either works, but Command Prompt is simpler

## Summary

```bash
# 1. Open Command Prompt

# 2. Navigate to project
cd Desktop\hand-sign-v3

# 3. Create virtual environment (first time only)
python -m venv venv

# 4. Activate virtual environment
venv\Scripts\activate

# 5. Install dependencies (first time only)
pip install -r requirements.txt

# 6. Run app
streamlit run app_yolo.py

# 7. Browser opens to http://localhost:8501
# 8. Click "Load Model" then "Start Camera"
# 9. Point hand at camera and make gestures!

# 10. Press Ctrl+C to stop when done
```

That's it! 🎉
