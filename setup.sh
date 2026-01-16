#!/bin/bash

# Hand Sign Detection - Quick Start Script

echo "🚀 Hand Sign Detection - Deployment Setup"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Verify model files
echo ""
echo "🔍 Checking model files..."
if [ -f "SSDLiteMobileNetV3Large.pth" ]; then
    echo "✓ Model file found"
else
    echo "✗ Warning: SSDLiteMobileNetV3Large.pth not found"
fi

if [ -f "hagrid-Hagrid_v2-1M/configs/SSDLiteMobileNetV3Large.yaml" ]; then
    echo "✓ Config file found"
else
    echo "✗ Warning: Config file not found"
fi

# Create .streamlit directory if not exists
mkdir -p .streamlit

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run locally:"
echo "  streamlit run app_web.py"
echo ""
echo "For camera version (local only):"
echo "  streamlit run app_yolo.py"
echo ""
echo "For Docker:"
echo "  docker build -t hand-sign-detection ."
echo "  docker run -p 8501:8501 hand-sign-detection"
echo ""
echo "For Render.com deployment:"
echo "  See DEPLOYMENT.md for detailed instructions"
