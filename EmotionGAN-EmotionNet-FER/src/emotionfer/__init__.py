"""EmotionGAN + EmotionNet-CNN reproducible implementation."""
__version__ = "1.0.0"
# Native FER-2013 label IDs: 0 Angry, 1 Disgust, 2 Fear, 3 Happy,
# 4 Sad, 5 Surprise, 6 Neutral (not the manuscript's textual order).
CLASSES = ("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral")
NUM_CLASSES = 7
