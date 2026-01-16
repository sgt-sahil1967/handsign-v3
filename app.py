"""
Streamlit app for Hand Sign Detection using HaGRID model
Based on hagrid-Hagrid_v2-1M/demo.py with continuous video feed
"""
import argparse
import logging
import os
import time
import warnings
from typing import Dict, Optional, Tuple

# Disable threading issues and SSL warnings BEFORE importing libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Suppress SSL certificate warnings
warnings.filterwarnings('ignore', message='.*Error fetching version info.*')
warnings.filterwarnings('ignore', category=UserWarning, module='.*albumentations.*')

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

# Lazy import MediaPipe to avoid threading issues
mp = None

def get_mediapipe():
    """Lazy load MediaPipe only when needed"""
    global mp
    if mp is None:
        try:
            import mediapipe as mp_module
            # Try to disable threading/GPU
            import os as mp_os
            mp_os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
            mp = mp_module
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load MediaPipe: {e}")
            mp = None
    return mp

# Disable OpenCV threading to prevent mutex conflicts
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Disable PyTorch threading (note: set_num_interop_threads must be called before Streamlit initializes)
try:
    torch.set_num_threads(1)
except:
    pass

# Import from hagrid module
import sys
sys.path.insert(0, 'hagrid-Hagrid_v2-1M')
from constants import targets
from custom_utils.utils import build_model

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
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background: #f0f2f6;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
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

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'transform' not in st.session_state:
    st.session_state.transform = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'landmarks_enabled' not in st.session_state:
    st.session_state.landmarks_enabled = False
if 'mp_hands' not in st.session_state:
    st.session_state.mp_hands = None


class Demo:
    """Demo class for hand sign detection - adapted from demo.py"""
    
    @staticmethod
    def preprocess(img: np.ndarray, transform) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Preproc image for model input
        """
        height, width = img.shape[0], img.shape[1]
        transformed_image = transform(image=img)
        processed_image = transformed_image["image"] / 255.0
        return processed_image, (width, height)

    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        """
        Create list of transforms from config
        """
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(
        detector,
        transform,
        conf: DictConfig,
        video_placeholder,
        status_placeholder,
        num_hands: int = 2,
        threshold: float = 0.5,
        landmarks: bool = False,
        stop_button = None
    ) -> None:
        """
        Run detection model and display results in Streamlit
        """
        cap = cv2.VideoCapture(0)
        st.session_state.video_capture = cap
        
        # Initialize MediaPipe hands if landmarks enabled
        hands = None
        mp_instance = None
        mp_drawing = None
        mp_drawing_styles = None
        landmarks_error = False
        
        if landmarks:
            try:
                mp_instance = get_mediapipe()
                # Disable MediaPipe's internal threading to avoid mutex conflicts
                import os as mp_os
                mp_os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
                
                mp_drawing = mp_instance.solutions.drawing_utils
                mp_drawing_styles = mp_instance.solutions.drawing_styles
                
                # Initialize hands detector with minimal threading
                hands = mp_instance.solutions.hands.Hands(
                    model_complexity=0,
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.8
                )
            except Exception as e:
                landmarks_error = True
                status_placeholder.warning(f"⚠️ Hand landmarks disabled due to threading conflict: {str(e)[:100]}")
                hands = None
                mp_instance = None
        else:
            mp_drawing = None
            mp_drawing_styles = None

        t1 = time.time()
        cnt = 0
        
        # Placeholder for FPS display
        fps_placeholder = st.empty()
        
        while st.session_state.running:
            if cap is None or not cap.isOpened():
                status_placeholder.error("Camera disconnected!")
                break

            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Failed to capture frame!")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Preprocess for model
            processed_image, size = Demo.preprocess(frame, transform)
            
            # Run inference
            with torch.no_grad():
                output = detector([processed_image])[0]
            
            boxes = output["boxes"][:num_hands]
            scores = output["scores"][:num_hands]
            labels = output["labels"][:num_hands]
            
            # Process hand landmarks if enabled (with error handling for threading issues)
            if landmarks and hands is not None and mp_drawing is not None:
                try:
                    results = hands.process(frame[:, :, ::-1])
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            try:
                                mp_drawing.draw_landmarks(
                                    display_frame,
                                    hand_landmarks,
                                    mp_instance.solutions.hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                                    mp_drawing_styles.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1),
                                )
                            except Exception as inner_e:
                                pass  # Silently skip if drawing fails
                except Exception as e:
                    # Silently continue if landmarks processing fails
                    pass
            
            # Draw bounding boxes
            detection_count = 0
            for i in range(min(num_hands, len(boxes))):
                if scores[i] > threshold:
                    detection_count += 1
                    width, height = size
                    scale = max(width, height) / conf.LongestMaxSize.max_size
                    padding_w = abs(conf.PadIfNeeded.min_width - width // scale) // 2
                    padding_h = abs(conf.PadIfNeeded.min_height - height // scale) // 2
                    
                    x1 = int((boxes[i][0] - padding_w) * scale)
                    y1 = int((boxes[i][1] - padding_h) * scale)
                    x2 = int((boxes[i][2] - padding_w) * scale)
                    y2 = int((boxes[i][3] - padding_h) * scale)
                    
                    # Ensure coordinates are within frame bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Only draw if box is valid
                    if x2 > x1 and y2 > y1:
                        # Draw rectangle with bright green color
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
                        
                        # Get gesture label
                        label_idx = int(labels[i])
                        gesture_name = targets.get(label_idx, f"unknown_{label_idx}")
                        
                        # Draw label background
                        label_size = cv2.getTextSize(gesture_name, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                        cv2.rectangle(display_frame, (x1, y1 - 25), (x1 + label_size[0], y1), (0, 255, 0), -1)
                        
                        # Draw label text in black on green background
                        cv2.putText(
                            display_frame,
                            gesture_name,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 0),
                            thickness=2,
                        )
                        
                        # Draw confidence score
                        conf_text = f"{scores[i]:.2f}"
                        cv2.putText(
                            display_frame,
                            conf_text,
                            (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            thickness=2,
                        )
            
            # Calculate and display FPS
            fps = 1 / delta if delta > 0 else 0
            
            # Create status text showing detections and threshold
            status_text = f"FPS: {fps:.1f} | Detections: {detection_count} | Threshold: {threshold:.2f} | Frame: {cnt}"
            
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
            
            # Display in Streamlit with fixed width for better performance
            video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update status with detection info
            if cnt % 3 == 0:  # Update every 3 frames to reduce overhead
                status_placeholder.write(
                    f"🎯 **Detections: {detection_count}** | 📊 Threshold: {threshold:.2f} | "
                    f"⚙️ FPS: {fps:.1f} | 📹 Frame: {cnt}"
                )
            
            # Update FPS display less frequently to reduce overhead
            if cnt % 5 == 0:
                fps_placeholder.metric("FPS", f"{fps:.1f}")
            
            # Small delay to allow UI updates
            time.sleep(0.01)
            
            # Check if stop button was pressed (via session state)
            if not st.session_state.running:
                break

        # Cleanup
        if cap is not None:
            cap.release()
        if hands is not None:
            hands.close()
        st.session_state.video_capture = None


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Demo detection...")
    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")
    parser.add_argument("-lm", "--landmarks", required=False, action="store_true", help="Use landmarks")
    
    known_args, _ = parser.parse_known_args(params)
    return known_args


@st.cache_resource
def load_detection_model(config_path: str, checkpoint_path: str = None):
    """Load the detection model with caching"""
    import warnings
    
    # Suppress deprecation warnings during model loading
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', message='.*pretrained.*')
        warnings.filterwarnings('ignore', message='.*weights.*')
        
        conf = OmegaConf.load(config_path)
        model = build_model(conf)
        transform = Demo.get_transform_for_inf(conf.test_transforms)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            snapshot = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(snapshot["MODEL_STATE"])
        
        model.eval()
    
    return model, transform, conf


def main():
    """Main application"""
    st.markdown('<div class="main-header">✋ Hand Sign Detection</div>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model configuration
        st.markdown("#### Model Config")
        config_path = st.text_input(
            "Config Path",
            value="hagrid-Hagrid_v2-1M/configs/SSDLiteMobileNetV3Large.yaml",
            help="Path to model configuration YAML file"
        )
        
        checkpoint_path = st.text_input(
            "Checkpoint Path (optional)",
            value="",
            help="Path to model checkpoint .pth file"
        )
        
        # Detection settings
        st.markdown("#### Detection Settings")
        num_hands = st.slider("Max Hands to Detect", 1, 10, 2)
        threshold = st.slider("Confidence Threshold", 0.001, 0.5, 0.01, step=0.001)
        
        # Landmarks toggle (disabled in Streamlit due to threading conflicts)
        landmarks_enabled = False  # Disabled due to threading issues with MediaPipe in Streamlit
        st.info("ℹ️ Hand landmarks feature is currently disabled due to threading conflicts between MediaPipe and Streamlit on macOS. Focus on gesture detection instead.")
        st.session_state.landmarks_enabled = landmarks_enabled
        
        st.markdown("---")
        st.markdown("#### About")
        st.markdown("""
        This app uses the HaGRID hand gesture detection model 
        to detect 34 different hand gestures in real-time.
        
        **Supported Gestures:**
        - Palm, Fist, OK, Peace
        - Like, Dislike
        - Stop, Rock
        - Call, Mute
        - And many more...
        """)
        
        st.markdown("---")
        st.markdown("#### Statistics")
        det_col1, det_col2 = st.columns(2)
        with det_col1:
            st.metric("Detections", 0, help="Number of gestures detected")
        with det_col2:
            st.metric("Threshold", f"{threshold:.3f}", help="Current confidence threshold")
        
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
                help="Load the detection model"
            )
        
        if load_btn:
            if not os.path.exists(config_path):
                status_placeholder.error(f"Config file not found: {config_path}")
            else:
                with st.spinner("Loading model... This may take a moment."):
                    try:
                        model, transform, config = load_detection_model(config_path, checkpoint_path if checkpoint_path else None)
                        st.session_state.model = model
                        st.session_state.transform = transform
                        st.session_state.config = config
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
                    Demo.run(
                        st.session_state.model,
                        st.session_state.transform,
                        st.session_state.config.test_transforms,
                        video_placeholder,
                        status_placeholder,
                        num_hands=num_hands,
                        threshold=threshold,
                        landmarks=st.session_state.landmarks_enabled
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
            st.info("👆 Click 'Load Model' to load the detection model, then click 'Start Camera' to begin.")
        
        # Display placeholder message
        if not st.session_state.running and st.session_state.model is not None:
            video_placeholder.info("👆 Click 'Start Camera' to begin detection")
    
    with col2:
        st.markdown("### 🎯 Detection Info")
        
        if st.session_state.model is not None:
            st.markdown("#### Model Status")
            st.success("✅ Model Loaded")
            
            st.markdown("#### Settings")
            st.markdown(f"**Max Hands:** {num_hands}")
            st.markdown(f"**Threshold:** {threshold:.2f}")
            st.markdown(f"**Landmarks:** {'Yes' if st.session_state.landmarks_enabled else 'No'}")
            
            st.markdown("---")
            st.markdown("#### Supported Gestures")
            st.markdown("""
            The model can detect 34 different hand gestures including:
            
            - Grabbing, Grip, Point
            - Call, Three, Timeout
            - Hand Heart, Like, Dislike
            - Fist, Four, Palm
            - Peace, Rock, Stop
            - And more...
            """)
        else:
            st.info("Load model to see detection info")


if __name__ == "__main__":
    main()

