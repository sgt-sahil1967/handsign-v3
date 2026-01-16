"""
Streamlit app for Hand Sign Detection using SSDLiteMobileNetV3Large model
"""
import argparse
import logging
import os
import sys
import time
import warnings
from typing import Dict, Optional, Tuple

# Add hagrid to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hagrid-Hagrid_v2-1M'))

# Disable threading before any imports
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', message='.*Error fetching version info.*')
warnings.filterwarnings('ignore', category=UserWarning, module='.*albumentations.*')

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf

# Import hagrid utilities - now should work since we added the path
try:
    from constants import targets
    from custom_utils.utils import build_model
except ImportError as e:
    st.error(f"Failed to import hagrid utilities: {e}")
    st.stop()

# Disable OpenCV threading
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Disable PyTorch threading
try:
    torch.set_num_threads(1)
except:
    pass

# Page configuration
st.set_page_config(
    page_title="Hand Sign Detection",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
logging.basicConfig(
    format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s",
    level=logging.INFO
)

# Constants
COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GESTURES = targets

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'transform' not in st.session_state:
    st.session_state.transform = None
if 'conf' not in st.session_state:
    st.session_state.conf = None


def get_transform_for_inference(transform_config: DictConfig):
    """Create list of transforms from config"""
    transforms_list = []
    for key, params in transform_config.items():
        if key == 'PadIfNeeded':
            # Fix PadIfNeeded parameters
            fixed_params = dict(params)
            if 'fill_value' in fixed_params:
                fixed_params.pop('fill_value')  # Remove unsupported parameter
            transforms_list.append(A.PadIfNeeded(**fixed_params))
        else:
            transforms_list.append(getattr(A, key)(**params))
    
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)


def preprocess(img: np.ndarray, transform) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess image for model input"""
    height, width = img.shape[0], img.shape[1]
    transformed_image = transform(image=img)
    processed_image = transformed_image["image"] / 255.0
    return processed_image, (width, height)


class YOLODemo:
    """Demo class for SSDLiteMobileNetV3Large hand sign detection"""
    
    @staticmethod
    def run(
        model,
        transform,
        conf,
        video_placeholder,
        status_placeholder,
        confidence: float = 0.5,
        num_hands: int = 100,
        stop_button = None
    ) -> None:
        """
        Run detection model and display results in Streamlit
        """
        # Validate inputs
        if model is None:
            status_placeholder.error("❌ Model is None!")
            return
        if transform is None:
            status_placeholder.error("❌ Transform is None!")
            return
        if conf is None:
            status_placeholder.error("❌ Config is None!")
            return
            
        cap = cv2.VideoCapture(0)
        st.session_state.video_capture = cap
        
        if not cap.isOpened():
            status_placeholder.error("❌ Failed to open camera!")
            return
        
        t1 = time.time()
        cnt = 0
        fps_placeholder = st.empty()
        
        while st.session_state.running:
            if cap is None or not cap.isOpened():
                status_placeholder.error("❌ Camera disconnected!")
                break

            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("❌ Failed to capture frame!")
                break
            
            display_frame = frame.copy()
            
            # Run inference
            try:
                processed_image, size = preprocess(frame, transform)
                with torch.no_grad():
                    output = model([processed_image])[0]
                
                boxes = output["boxes"][:num_hands]
                scores = output["scores"][:num_hands]
                labels = output["labels"][:num_hands]
                
                # Convert tensors to numpy/python types for easier handling
                if torch.is_tensor(boxes):
                    boxes = boxes.cpu().numpy()
                if torch.is_tensor(scores):
                    scores = scores.cpu().numpy()
                if torch.is_tensor(labels):
                    labels = labels.cpu().numpy()
                
                detection_count = 0
                
                for i in range(min(num_hands, len(boxes))):
                    if scores[i] > confidence:
                        detection_count += 1
                        width, height = size
                        scale = max(width, height) / conf.test_transforms.LongestMaxSize.max_size
                        padding_w = abs(conf.test_transforms.PadIfNeeded.min_width - width // scale) // 2
                        padding_h = abs(conf.test_transforms.PadIfNeeded.min_height - height // scale) // 2
                        
                        x1 = int((boxes[i][0] - padding_w) * scale)
                        y1 = int((boxes[i][1] - padding_h) * scale)
                        x2 = int((boxes[i][2] - padding_w) * scale)
                        y2 = int((boxes[i][3] - padding_h) * scale)
                        
                        # Ensure coordinates are within frame
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Get gesture name
                        gesture_name = GESTURES[int(labels[i])]
                        conf_score = float(scores[i])
                        
                        # Draw label with background
                        label_text = f"{gesture_name} {conf_score:.2f}"
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + label_size[0], y1), (0, 255, 0), -1)
                        cv2.putText(
                            display_frame,
                            label_text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 0),
                            2,
                        )
            except Exception as e:
                status_placeholder.warning(f"⚠️ Inference error: {str(e)[:200]}")
                print(f"Full error: {e}")
                import traceback
                traceback.print_exc()
                detection_count = 0
            
            # Calculate and display FPS
            fps = 1 / delta if delta > 0 else 0
            
            # Create status text
            status_text = f"FPS: {fps:.1f} | Detections: {detection_count} | Confidence: {confidence:.2f}"
            
            cv2.putText(
                display_frame,
                status_text,
                (30, 30),
                FONT,
                0.7,
                (0, 255, 0),
                2
            )
            cnt += 1
            
            # Convert to RGB for display
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Display in Streamlit
            video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update status with detection info
            if cnt % 3 == 0:
                status_placeholder.write(
                    f"🎯 **Detections: {detection_count}** | 📊 Confidence: {confidence:.2f} | "
                    f"⚙️ FPS: {fps:.1f} | 📹 Frame: {cnt}"
                )
            
            # Update FPS display
            if cnt % 5 == 0:
                fps_placeholder.metric("FPS", f"{fps:.1f}")
            
            # Small delay to allow UI updates
            time.sleep(0.01)
            
            # Check if stop button was pressed
            if not st.session_state.running:
                break

        # Cleanup
        if cap is not None:
            cap.release()
        st.session_state.video_capture = None


@st.cache_resource
def load_ssd_model(model_path: str, config_path: str):
    """Load SSDLiteMobileNetV3Large model with caching"""
    try:
        print(f"Loading config from: {config_path}")
        conf = OmegaConf.load(config_path)
        print(f"Config loaded successfully")
        
        print(f"Building model...")
        model = build_model(conf)
        print(f"Model built successfully")
        
        print(f"Loading checkpoint from: {model_path}")
        snapshot = torch.load(model_path, map_location=torch.device("cpu"))
        print(f"Checkpoint loaded. Keys: {snapshot.keys()}")
        
        print(f"Loading state dict...")
        model.load_state_dict(snapshot["MODEL_STATE"])
        print(f"State dict loaded")
        
        model.eval()
        print(f"Model set to eval mode")
        
        transform = get_transform_for_inference(conf.test_transforms)
        print(f"Transform created")
        
        print(f"Model loaded successfully!")
        return model, transform, conf
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        st.error(f"Failed to load model: {e}")
        return None, None, None


def main():
    """Main application"""
    st.markdown('<div class="main-header">✋ Hand Sign Detection (SSDLiteMobileNetV3Large)</div>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model configuration
        st.markdown("#### Model Config")
        model_path = "SSDLiteMobileNetV3Large.pth"
        config_path = "hagrid-Hagrid_v2-1M/configs/SSDLiteMobileNetV3Large.yaml"
        
        st.write(f"**Model:** {model_path}")
        st.write(f"**Config:** {config_path}")
        
        # Detection settings
        st.markdown("#### Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)
        num_hands = st.slider("Max Hands to Detect", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("#### Statistics")
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            st.metric("Detections", 0, help="Number of gestures detected")
        with det_col2:
            st.metric("Confidence", f"{confidence:.2f}", help="Current confidence threshold")
        
        st.markdown("---")
        st.markdown("#### Supported Gestures")
        gesture_cols = st.columns(2)
        for i, gesture in enumerate(GESTURES):
            col = gesture_cols[i % 2]
            col.write(f"• {gesture}")
        
        st.markdown("---")
        st.markdown("#### FPS")
        fps_display = st.empty()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎥 Live Camera Feed")
        
        # Video placeholder
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Control buttons
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            load_btn = st.button(
                "📦 Load Model",
                type="primary",
                use_container_width=True,
                help="Load the SSDLiteMobileNetV3Large model"
            )
        
        if load_btn:
            if not os.path.exists(model_path):
                status_placeholder.error(f"❌ Model file not found: {model_path}")
            elif not os.path.exists(config_path):
                status_placeholder.error(f"❌ Config file not found: {config_path}")
            else:
                with st.spinner("Loading model... This may take a moment."):
                    try:
                        model, transform, conf = load_ssd_model(model_path, config_path)
                        st.session_state.model = model
                        st.session_state.transform = transform
                        st.session_state.conf = conf
                        status_placeholder.success("✅ Model loaded successfully!")
                    except Exception as e:
                        status_placeholder.error(f"Error loading model: {e}")
        
        with col_btn2:
            if not st.session_state.running:
                start_btn = st.button(
                    "▶️ Start Camera",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.model is None,
                    help="Start the camera and detection"
                )
                
                if start_btn and st.session_state.model is not None:
                    st.session_state.running = True
                    YOLODemo.run(
                        st.session_state.model,
                        st.session_state.transform,
                        st.session_state.conf,
                        video_placeholder,
                        status_placeholder,
                        confidence=confidence,
                        num_hands=num_hands,
                    )
            else:
                stop_btn = st.button(
                    "⏹️ Stop Camera",
                    type="secondary",
                    use_container_width=True,
                    help="Stop the camera and detection"
                )
                if stop_btn:
                    st.session_state.running = False
                    if st.session_state.video_capture is not None:
                        st.session_state.video_capture.release()
                        st.session_state.video_capture = None
                    time.sleep(0.5)
                    st.rerun()
        
        # Instructions
        if st.session_state.model is None:
            st.info("👆 Click 'Load Model' to load the SSDLiteMobileNetV3Large model, then click 'Start Camera' to begin.")
        
        # Display placeholder message
        if not st.session_state.running and st.session_state.model is not None:
            video_placeholder.info("👆 Click 'Start Camera' to begin detection")
    
    with col2:
        st.markdown("### 📊 Info")
        if st.session_state.model is not None:
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Load Model First")


if __name__ == "__main__":
    main()
