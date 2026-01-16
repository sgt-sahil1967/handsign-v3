# Hand Sign Detection - Render.com Deployment Guide

## Overview
This app has been prepared for deployment on Render.com with the following components:
- `app_web.py` - Web version with image upload support
- `requirements.txt` - All Python dependencies
- `render.yaml` - Render deployment configuration
- `.streamlit/config.toml` - Streamlit configuration

## Local Testing

### Prerequisites
- Python 3.9+
- pip or conda

### Setup
```bash
pip install -r requirements.txt
```

### Running Locally
```bash
# For web version (recommended for deployment testing)
streamlit run app_web.py

# For local camera version
streamlit run app_yolo.py
```

## Deployment to Render.com

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit for Render deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Create Render Account
1. Go to https://render.com
2. Sign up for a free account
3. Connect your GitHub account

### Step 3: Deploy
1. Click "New +" → "Web Service"
2. Select your repository
3. Configure:
   - **Name**: `hand-sign-detection`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app_web.py --server.port=$PORT --server.address=0.0.0.0`
   - **Free Tier** or **Paid Plan** (Free tier is fine for testing)
4. Click "Create Web Service"

### Step 4: Configure Environment
In Render dashboard, go to Settings → Environment:
```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=$PORT
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## Features

### Web Version (app_web.py)
✅ Image upload support
✅ Works on Render.com and other cloud platforms
✅ No camera access required
✅ Detects hand gestures in uploaded images

### Local Version (app_yolo.py)
✅ Real-time camera detection
✅ Live video stream
✅ For local/desktop use only

## Model Information
- **Model**: SSDLiteMobileNetV3Large
- **Framework**: PyTorch
- **Input Size**: 224x224
- **Supported Gestures**: ~50+ hand signs/gestures

## Troubleshooting

### Model Loading Issues
- Ensure `SSDLiteMobileNetV3Large.pth` is in the root directory
- Check that config file exists at `hagrid-Hagrid_v2-1M/configs/SSDLiteMobileNetV3Large.yaml`

### Memory Issues on Render
- Upgrade to a paid plan with more memory
- Or use a smaller model variant if available

### Slow Inference
- This is expected on free Render tier due to limited CPU
- Upgrade to a paid plan or use GPU options if available

## API Endpoints
The app exposes a Streamlit interface at:
- Local: `http://localhost:8501`
- Render: `https://your-app-name.onrender.com`

## Performance Notes
- First inference may take 2-5 seconds (model loading)
- Subsequent inferences are faster due to caching
- Works best with images 640x480 or similar resolution

## Support
For issues, check the Render logs:
```
Render Dashboard → Your App → Logs
```

## Additional Resources
- Render Documentation: https://render.com/docs
- Streamlit Documentation: https://docs.streamlit.io
- PyTorch Documentation: https://pytorch.org/docs
