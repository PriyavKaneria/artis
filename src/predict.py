# src/predict.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt # Still used for dataset mode display
import os
import random
import glob
import cv2 # For image processing

# --- PyQt5 Imports ---
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGraphicsView, QGraphicsScene, QSizePolicy, QGraphicsPathItem)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap, QPolygonF, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QRectF

# --- Import custom layer from model_def ---
from model_def import FiLMLayer # CRUCIAL for loading the model

# --- Configuration ---
IMG_DIM = 128 # Model's expected input dimension
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
MODEL_PATH = "trained_models/cat_generator_film_v2.keras"

# Paths from your previous file structure
OUTPUT_DIR_BASE_PROCESSED = "dataset/processed/grayscale_augmented_cat_v2/"
GRAYSCALE_CATS_DIR_SAMPLING = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "grayscale_cats/")
OUTLINE_MASKS_DIR_SAMPLING = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "outline_masks/")


# --- Mode Selection ---
USE_DATASET_OUTLINES = True
USE_DRAWING_CANVAS = False

# --- Dataset Mode Configuration ---
NUM_DATASET_TEST_ITERATIONS = 3

# --- Drawing Mode Configuration ---
DRAWING_CANVAS_SIZE = 512
PREVIEW_IMAGE_SIZE = 64

print(f"Sampling grayscale from: {GRAYSCALE_CATS_DIR_SAMPLING}")
print(f"Sampling outlines from: {OUTLINE_MASKS_DIR_SAMPLING}")

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first.")
try:
    # When loading a model with custom layers, provide them in custom_objects
    tf.keras.config.enable_unsafe_deserialization()
    trained_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'FiLMLayer': FiLMLayer}
    )
    print("Trained FiLM model loaded successfully.")
except Exception as e:
    print(f"Error loading FiLM model: {e}")
    print("Ensure 'FiLMLayer' is correctly defined in model_def.py and passed to custom_objects.")
    print("If saved in Keras native format, try loading without .h5 extension and ensure TensorFlow version compatibility.")
    exit()

# --- Helper Functions (mostly same as before) ---
def get_base_image_paths(primary_data_dir, outline_dir, primary_suffix="_grayscale.npy"):
    primary_files_pattern = os.path.join(primary_data_dir, f"*{primary_suffix}")
    all_primary_file_paths = sorted(glob.glob(primary_files_pattern))
    base_primary_paths = []
    # For outlines, we want to pair with the primary data's base names,
    # but the outline files themselves might have _augX_approxY in their names.
    # The key is that the grayscale file for examples must be a "base" one (no _aug, no _approx).
    
    for p_fpath in all_primary_file_paths:
        # We only want *base* grayscale images for the example pool
        if "_aug" not in os.path.basename(p_fpath) and "_approx" not in os.path.basename(p_fpath):
            base_primary_paths.append(p_fpath)
            
    # For dataset mode, we might want to test with various outlines (base, aug, approx)
    # but pair them with their corresponding grayscale content if it exists, or base content.
    # The current logic for dataset_prediction_mode handles loading pairs.
    # For now, just return base grayscale paths for example selection.
    # The outline paths for dataset mode are derived differently there.
    all_outline_paths = sorted(glob.glob(os.path.join(outline_dir, "*_outline.npy")))
    return base_primary_paths, all_outline_paths


BASE_GRAYSCALE_PATHS_FOR_EXAMPLES, ALL_AVAILABLE_OUTLINE_PATHS_FOR_DATASET_MODE = get_base_image_paths(
    GRAYSCALE_CATS_DIR_SAMPLING,
    OUTLINE_MASKS_DIR_SAMPLING,
    primary_suffix="_grayscale.npy"
)
if not BASE_GRAYSCALE_PATHS_FOR_EXAMPLES:
    print("Warning: No base grayscale images found for examples. Example selection might fail.")


def load_images_from_paths(paths_list, img_shape_tuple):
    images = []
    for path in paths_list:
        try:
            img = np.load(path).reshape(img_shape_tuple)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load or reshape image {path}: {e}")
    return images

def np_to_qpixmap(np_array, target_size=None, is_mask=False):
    if np_array.ndim == 3 and np_array.shape[2] == 1:
        np_array = np_array.squeeze()
    if np_array.dtype == np.float32 or np_array.dtype == np.float64:
        np_array = (np_array * 255).astype(np.uint8)
    elif np_array.dtype != np.uint8:
        np_array = np_array.astype(np.uint8)
    height, width = np_array.shape
    bytes_per_line = width
    q_image = QImage(np_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(q_image)
    if target_size:
        pixmap = pixmap.scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap

def predict_single(model, example_primary_imgs_list, input_outline_mask_single):
    if len(example_primary_imgs_list) != NUM_EXAMPLES_CONDITION:
        print(f"Error: Expected {NUM_EXAMPLES_CONDITION} example images, got {len(example_primary_imgs_list)}")
        return None
    batch_example_inputs = [np.expand_dims(img, axis=0) for img in example_primary_imgs_list]
    batch_outline_input_for_generator = np.expand_dims(input_outline_mask_single, axis=0)
    model_inputs_for_pred = batch_example_inputs + [batch_outline_input_for_generator]
    generated_batch = model.predict(model_inputs_for_pred)
    return generated_batch[0]


# --- Dataset Outline Mode (using Matplotlib for display) ---
def dataset_prediction_mode():
    print("\n--- Starting Dataset Outline Prediction Mode ---")
    if not BASE_GRAYSCALE_PATHS_FOR_EXAMPLES:
        print("Error: No base grayscale images found for examples. Cannot proceed.")
        return
    if not ALL_AVAILABLE_OUTLINE_PATHS_FOR_DATASET_MODE:
        print("Error: No outline masks found in dataset directory. Cannot proceed.")
        return

    num_tests = min(NUM_DATASET_TEST_ITERATIONS, len(ALL_AVAILABLE_OUTLINE_PATHS_FOR_DATASET_MODE))
    if len(ALL_AVAILABLE_OUTLINE_PATHS_FOR_DATASET_MODE) < NUM_DATASET_TEST_ITERATIONS:
        print(f"Warning: Requested {NUM_DATASET_TEST_ITERATIONS} tests, but only {len(ALL_AVAILABLE_OUTLINE_PATHS_FOR_DATASET_MODE)} outlines available.")

    # selected_outline_paths_for_test = random.sample(ALL_AVAILABLE_OUTLINE_PATHS_FOR_DATASET_MODE, num_tests)
    # Fixed outline image
    selected_outline_paths_for_test = ["dataset/processed/grayscale_augmented_cat_v2/outline_masks/52_outline.npy"]

    for i, input_outline_path in enumerate(selected_outline_paths_for_test):
        print(f"\n--- Test Iteration {i+1}/{num_tests} ---")
        
        input_outline_img_list = load_images_from_paths([input_outline_path], IMG_SHAPE)
        if not input_outline_img_list: 
            print(f"Failed to load input outline: {input_outline_path}")
            continue
        input_outline_img = input_outline_img_list[0]

        # Select example grayscale images (always from base non-augmented, non-approx)
        num_examples_to_sample = min(NUM_EXAMPLES_CONDITION, len(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES))
        if num_examples_to_sample < NUM_EXAMPLES_CONDITION:
             print(f"Warning: Using {num_examples_to_sample} examples as only that many base grayscales are available.")

        selected_example_paths = random.sample(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES, num_examples_to_sample)
        example_images = load_images_from_paths(selected_example_paths, IMG_SHAPE)

        if len(example_images) != num_examples_to_sample:
            print(f"Failed to load sufficient example images for outline {os.path.basename(input_outline_path)}")
            continue
            
        print(f"Using outline: {os.path.basename(input_outline_path)}")
        print(f"Example images: {[os.path.basename(p) for p in selected_example_paths]}")

        generated_img = predict_single(trained_model, example_images, input_outline_img)
        if generated_img is not None:
            plt.figure(figsize=(3 * (num_examples_to_sample + 2), 3) ) # Dynamic figure size
            for j_ex, ex_img in enumerate(example_images):
                plt.subplot(1, num_examples_to_sample + 2, j_ex + 1)
                plt.imshow(ex_img.squeeze(), cmap='gray', vmin=0, vmax=1); plt.title(f"Ex {j_ex+1}"); plt.axis('off')
            plt.subplot(1, num_examples_to_sample + 2, num_examples_to_sample + 1)
            plt.imshow(input_outline_img.squeeze(), cmap='gray_r'); plt.title("Input Outline"); plt.axis('off')
            plt.subplot(1, num_examples_to_sample + 2, num_examples_to_sample + 2)
            plt.imshow(generated_img.squeeze(), cmap='gray', vmin=0, vmax=1); plt.title("Generated"); plt.axis('off')
            plt.suptitle(f"Dataset Prediction {i+1}"); plt.tight_layout(rect=[0,0,1,0.93]); plt.show()
    print("\n--- Dataset Outline Prediction Mode Finished ---")


# --- PyQt5 Drawing Canvas Mode (largely same as your previous version) ---
class DrawingScene(QGraphicsScene): # (Same as your previous PyQt DrawingScene)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_path = None
        self.points = []
        self.is_drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            # For a new stroke, if current_path exists, add it permanently before starting new one
            # Or, assume each 'g' processes the current state, and 'c' clears all.
            # Simplest: each drag is one continuous path part.
            if not self.current_path or not self.is_drawing: # Start new path object
                self.current_path = QPainterPath()
                self.current_path.moveTo(event.scenePos())
                self.points = [event.scenePos()]
            else: # Continue existing path
                self.current_path.lineTo(event.scenePos())
                self.points.append(event.scenePos())
            self.update_path_item() # Add or update visual item for the path
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing and (event.buttons() & Qt.LeftButton):
            if self.current_path:
                self.current_path.lineTo(event.scenePos())
                self.points.append(event.scenePos())
                self.update_path_item()
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            # Path is already drawn by mouseMoveEvent's update_path_item
            super().mouseReleaseEvent(event)
            
    def update_path_item(self):
        # Instead of removing all, manage one persistent path item or multiple if allowing separate strokes
        # For simplicity, assume one continuous path object that gets refined
        items_to_remove = [item for item in self.items() if isinstance(item, QGraphicsPathItem)]
        for item in items_to_remove:
            self.removeItem(item)
        
        if self.current_path:
            pen = QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin) # Thicker pen
            self.addPath(self.current_path, pen)

    def clear_scene(self):
        self.points = []
        self.current_path = None
        items_to_remove = [item for item in self.items() if isinstance(item, QGraphicsPathItem)]
        for item in items_to_remove:
            self.removeItem(item)
        self.update()

    def get_drawn_mask(self, target_dim):
        if not self.points and not self.current_path: return None

        image = QImage(DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.white)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Render current path if it exists
        if self.current_path:
            pen = QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(self.current_path)
        
        # Attempt to close if open (heuristic)
        if self.points and len(self.points) > 2:
            start_qpoint, end_qpoint = self.points[0], self.points[-1]
            if (start_qpoint.x() - end_qpoint.x())**2 + (start_qpoint.y() - end_qpoint.y())**2 > 15**2:
                closing_path = QPainterPath()
                closing_path.moveTo(end_qpoint)
                closing_path.lineTo(start_qpoint)
                painter.drawPath(closing_path)
        painter.end()

        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4) # BGRA
        gray_arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        
        inverted_gray = cv2.bitwise_not(gray_arr)
        contours, _ = cv2.findContours(inverted_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_mask_512 = np.ones((DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE), dtype=np.uint8) * 255
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled_mask_512, [largest_contour], -1, 0, thickness=cv2.FILLED)
        
        scaled_mask_cv = cv2.resize(filled_mask_512, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)
        model_input_mask_np = (scaled_mask_cv.astype(np.float32) / 255.0).reshape((target_dim, target_dim, 1))
        return model_input_mask_np, scaled_mask_cv

class DrawingWindow(QWidget): # (Same as your previous PyQt DrawingWindow)
    def __init__(self):
        super().__init__()
        self.example_model_inputs = []
        self.initUI()
        self.load_example_images()

    def initUI(self):
        self.setWindowTitle('Interactive Cat Generation (FiLM Model)')
        self.setGeometry(100, 100, DRAWING_CANVAS_SIZE + IMG_DIM*2 + 100, DRAWING_CANVAS_SIZE + 150)
        main_layout = QVBoxLayout(); self.setLayout(main_layout)
        self.examples_layout = QHBoxLayout()
        self.example_labels = [QLabel(f"Ex {i+1}") for i in range(NUM_EXAMPLES_CONDITION)]
        for label in self.example_labels:
            label.setFixedSize(PREVIEW_IMAGE_SIZE, PREVIEW_IMAGE_SIZE); label.setStyleSheet("border: 1px solid gray;"); label.setAlignment(Qt.AlignCenter); self.examples_layout.addWidget(label)
        main_layout.addLayout(self.examples_layout)
        drawing_results_layout = QHBoxLayout(); main_layout.addLayout(drawing_results_layout)
        self.scene = DrawingScene(); self.scene.setSceneRect(0, 0, DRAWING_CANVAS_SIZE -2, DRAWING_CANVAS_SIZE -2)
        self.view = QGraphicsView(self.scene); self.view.setFixedSize(DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE); self.view.setBackgroundBrush(Qt.white)
        self.view.setRenderHint(QPainter.Antialiasing); self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        drawing_results_layout.addWidget(self.view)
        self.input_mask_label = QLabel("Input Mask"); self.input_mask_label.setFixedSize(IMG_DIM, IMG_DIM); self.input_mask_label.setStyleSheet("border: 1px solid blue;"); self.input_mask_label.setAlignment(Qt.AlignCenter); drawing_results_layout.addWidget(self.input_mask_label)
        self.generated_image_label = QLabel("Generated Image"); self.generated_image_label.setFixedSize(IMG_DIM, IMG_DIM); self.generated_image_label.setStyleSheet("border: 1px solid green;"); self.generated_image_label.setAlignment(Qt.AlignCenter); drawing_results_layout.addWidget(self.generated_image_label)
        buttons_layout = QHBoxLayout(); self.generate_button = QPushButton("Generate (g)"); self.generate_button.clicked.connect(self.on_generate)
        self.clear_button = QPushButton("Clear (c)"); self.clear_button.clicked.connect(self.on_clear)
        buttons_layout.addWidget(self.generate_button); buttons_layout.addWidget(self.clear_button); main_layout.addLayout(buttons_layout)
        self.show()

    def load_example_images(self):
        if not BASE_GRAYSCALE_PATHS_FOR_EXAMPLES: return
        num_to_sample = min(NUM_EXAMPLES_CONDITION, len(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES))
        selected_example_paths = random.sample(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES, num_to_sample)
        self.example_model_inputs = load_images_from_paths(selected_example_paths, IMG_SHAPE)
        for i in range(NUM_EXAMPLES_CONDITION):
            if i < len(self.example_model_inputs): self.example_labels[i].setPixmap(np_to_qpixmap(self.example_model_inputs[i], PREVIEW_IMAGE_SIZE))
            else: self.example_labels[i].clear(); self.example_labels[i].setText(f"Ex {i+1} N/A")

    def on_generate(self):
        print("Generate triggered.")
        if not self.example_model_inputs or len(self.example_model_inputs) != NUM_EXAMPLES_CONDITION:
            print("Not enough example images loaded."); self.generated_image_label.setText("Load Examples!"); return
        mask_data = self.scene.get_drawn_mask(IMG_DIM)
        if mask_data is None: print("No outline drawn."); self.input_mask_label.setText("Draw Outline!"); return
        model_input_mask_np, display_mask_cv = mask_data
        self.input_mask_label.setPixmap(np_to_qpixmap(display_mask_cv, IMG_DIM, is_mask=True)) # display_mask_cv is 0=obj, 255=bg
        generated_img_np = predict_single(trained_model, self.example_model_inputs, model_input_mask_np)
        if generated_img_np is not None:
            self.generated_image_label.setPixmap(np_to_qpixmap(generated_img_np, IMG_DIM)); print("Generation complete.")
        else: self.generated_image_label.setText("Generation Failed"); print("Generation failed.")

    def on_clear(self): self.scene.clear_scene(); self.input_mask_label.setText("Input Mask"); self.input_mask_label.clear(); self.generated_image_label.setText("Generated Image"); self.generated_image_label.clear()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_G: self.on_generate()
        elif event.key() == Qt.Key_C: self.on_clear()
        elif event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape: self.close()

def qt_drawing_mode():
    app = QApplication.instance() # Check if an instance already exists
    if app is None: # Create a new instance if it doesn't exist
        app = QApplication(sys.argv)
    
    # Check if Python is running in an environment that supports GUI interaction
    # This is a common check for scripts that might run in headless environments
    if not app.platformName() or app.platformName() == "offscreen" or app.platformName() == "minimal":
        print("Warning: PyQt5 may not be able to create a window in this environment.")
        print("If running in a headless server or a restricted environment, GUI mode will not work.")
        # Fallback or exit if GUI is essential
        # return # Or raise an error

    ex_window = DrawingWindow()
    sys.exit(app.exec_())


# --- Main Execution ---
def main():
    if USE_DATASET_OUTLINES and USE_DRAWING_CANVAS: print("Error: Both modes True. Choose one."); return
    if not USE_DATASET_OUTLINES and not USE_DRAWING_CANVAS: print("Error: No mode selected."); return
    if USE_DATASET_OUTLINES: dataset_prediction_mode()
    if USE_DRAWING_CANVAS:
        if not BASE_GRAYSCALE_PATHS_FOR_EXAMPLES: print("CRITICAL: No base grayscales for drawing examples. Exiting."); return
        qt_drawing_mode()

if __name__ == "__main__":
    main()