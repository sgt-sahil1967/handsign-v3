"""
Streamlit app for Hand Sign Detection using SSDLiteMobileNetV3Large model
Web version - supports image upload and file input
"""
import sys
import os
import time
import warnings
from typing import Tuple

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
from PIL import Image

# Import hagrid utilities
try:
    from constants import targets
    from custom_utils.utils import build_model
except ImportError as e:
    st.error(f"Failed to import hagrid utilities: {e}")
    st.stop()

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

# Constants
COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
GESTURES = targets

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'transform' not in st.session_state:
    st.session_state.transform = None
if 'conf' not in st.session_state:
    st.session_state.conf = None


def get_transform_for_inference(transform_config: DictConfig):
    """Create list of transforms from config"""
    transforms_list = []
    for key, params in transform_config.items():
        if key == 'PadIfNeeded':
            fixed_params = dict(params)
            if 'fill_value' in fixed_params:
                fixed_params.pop('fill_value')
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


def run_inference(image_array: np.ndarray, model, transform, conf, confidence: float = 0.5, num_hands: int = 100):
    """Run inference on a single image and return annotated image with detections"""
    display_frame = image_array.copy()
    
    try:
        if model is None or transform is None or conf is None:
            st.error("❌ Model not loaded properly!")
            return display_frame, 0
        
        # Preprocess
        processed_image, size = preprocess(image_array, transform)
        
        with torch.no_grad():
            output = model([processed_image])[0]
        
        boxes = output["boxes"][:num_hands]
        scores = output["scores"][:num_hands]
        labels = output["labels"][:num_hands]
        
        # Convert tensors to numpy/python types
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
                x2, y2 = min(image_array.shape[1], x2), min(image_array.shape[0], y2)
                
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
        
        return display_frame, detection_count
        
    except Exception as e:
        st.error(f"⚠️ Inference error: {str(e)[:200]}")
        print(f"Full error: {e}")
        import traceback
        traceback.print_exc()
        return display_frame, 0


def main():
    """Main application"""
    st.markdown('<div class="main-header">✋ Hand Sign Detection (Web Version)</div>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model configuration
        st.markdown("#### Model Config")
        model_path = "SSDLiteMobileNetV3Large.pth"
        config_path = "hagrid-Hagrid_v2-1M/configs/SSDLiteMobileNetV3Large.yaml"
        
        # Detection settings
        st.markdown("#### Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, step=0.05)
        num_hands = st.slider("Max Hands to Detect", 1, 10, 5)
        
        st.markdown("---")
        st.markdown("#### Supported Gestures")
        gesture_cols = st.columns(2)
        for i, gesture in enumerate(GESTURES):
            col = gesture_cols[i % 2]
            col.write(f"• {gesture}")
        
        # Load model button
        st.markdown("---")
        if st.button("📦 Load Model", use_container_width=True, type="primary"):
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found: {model_path}")
            elif not os.path.exists(config_path):
                st.error(f"❌ Config file not found: {config_path}")
            else:
                with st.spinner("Loading model... This may take a moment."):
                    model, transform, conf = load_ssd_model(model_path, config_path)
                    st.session_state.model = model
                    st.session_state.transform = transform
                    st.session_state.conf = conf
                    if model is not None:
                        st.success("✅ Model loaded successfully!")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            if st.session_state.model is None:
                st.warning("⚠️ Please load the model first using the sidebar button")
            else:
                # Run inference
                with st.spinner("Running inference..."):
                    result_frame, detection_count = run_inference(
                        image_array,
                        st.session_state.model,
                        st.session_state.transform,
                        st.session_state.conf,
                        confidence=confidence,
                        num_hands=num_hands,
                    )
                
                # Convert BGR to RGB for display
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Display results
                st.image(result_frame_rgb, use_container_width=True, caption=f"Detections: {detection_count}")
                
                # Show stats
                st.success(f"✅ Found {detection_count} hand gesture(s)")
        else:
            if st.session_state.model is not None:
                st.success("✅ Model Ready")
            else:
                st.info("👆 Load the model and upload an image to begin detection")
    
    with col2:
        st.markdown("### 📊 Info")
        if st.session_state.model is not None:
            st.success("✅ Model Loaded")
        else:
            st.warning("⚠️ Model Not Loaded")


if __name__ == "__main__":
    main()
