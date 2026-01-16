# Hand Sign Detection App

A modern Streamlit web application for real-time hand gesture detection using YOLOv10 model trained on HaGRID dataset.

## Features

- 🔐 JSON-based authentication system (mock login)
- 📷 Real-time hand sign detection via webcam
- 📁 Image upload and detection
- 🎨 Modern and professional UI
- ✋ Support for 18+ hand gestures

## Prerequisites

- Python 3.8 or higher
- YOLOv10x_gestures.pt model file (should be in the project root)

## Installation

1. Clone or navigate to this directory:
```bash
cd hand-sign-v3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure the model file `YOLOv10x_gestures.pt` is in the project root directory.

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## Default Login Credentials

The app comes with two pre-configured accounts:

- **Username:** `admin` | **Password:** `admin123`
- **Username:** `user` | **Password:** `user123`

You can also register a new account using the registration tab.

## Supported Gestures

The model can detect the following hand gestures:
- call, dislike, fist, four, like, mute, ok, one, palm
- peace, peace_inverted, rock, stop, stop_inverted
- three, three2, two_up, two_up_inverted

## Usage

1. **Login**: Use the provided credentials or register a new account
2. **Camera Detection**: Click the camera button to take a photo or use live camera feed
3. **Upload Image**: Alternatively, upload an image file (JPG, PNG, JPEG)
4. **View Results**: Detected gestures will be displayed with confidence scores

## Project Structure

```
hand-sign-v3/
├── app.py                 # Main Streamlit application
├── auth.py                # Authentication module
├── users.json             # User database (JSON)
├── requirements.txt       # Python dependencies
├── YOLOv10x_gestures.pt  # Pre-trained model
└── README.md             # This file
```

## Notes

- The authentication system is for demonstration purposes only (not secure for production)
- First model load may take a few seconds
- Make sure you have a webcam connected if using camera input
- The model requires sufficient lighting for best results

## Troubleshooting

- **Model not found**: Ensure `YOLOv10x_gestures.pt` is in the project root
- **Camera not working**: Check camera permissions in your browser
- **Slow detection**: First inference may be slower; subsequent detections should be faster

## License

This project uses the HaGRID dataset and YOLOv10 model. Please refer to their respective licenses.
