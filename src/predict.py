# src/predict.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob
import cv2 # Still used for image processing, not for drawing GUI

# --- PyQt5 Imports ---
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGraphicsView, QGraphicsScene, QSizePolicy, QGraphicsPathItem)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap, QPolygonF, QPainterPath
from PyQt5.QtCore import Qt, QPointF, QRectF


# --- Configuration ---
IMG_DIM = 128 # Model's expected input dimension
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
MODEL_PATH = "trained_models/cat_grayscale_augmented_generator.keras" # Preserved path

BASE_PROCESSED_DATA_DIR_FOR_SAMPLING = "dataset/processed/grayscale_augmented_cat/" # Preserved path
GRAYSCALE_CATS_DIR_SAMPLING = os.path.join(BASE_PROCESSED_DATA_DIR_FOR_SAMPLING, "grayscale_cats/")
OUTLINE_MASKS_DIR_SAMPLING = os.path.join(BASE_PROCESSED_DATA_DIR_FOR_SAMPLING, "outline_masks/")

# --- Mode Selection ---
USE_DATASET_OUTLINES = False
USE_DRAWING_CANVAS = True # Assuming True for this PyQt version

# --- Dataset Mode Configuration ---
NUM_DATASET_TEST_ITERATIONS = 3

# --- Drawing Mode Configuration ---
DRAWING_CANVAS_SIZE = 512 # User draws on this resolution
PREVIEW_IMAGE_SIZE = 64  # For displaying example images

print(f"Sampling grayscale from: {GRAYSCALE_CATS_DIR_SAMPLING}")
print(f"Sampling outlines from: {OUTLINE_MASKS_DIR_SAMPLING}")

# --- Load Model ---
# (Keep your existing model loading logic)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first.")
try:
    trained_model = tf.keras.models.load_model(MODEL_PATH)
    print("Trained model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Helper Functions (mostly same as before, adapted for PyQt where needed) ---
def get_base_image_paths(primary_data_dir, outline_dir, primary_suffix="_grayscale.npy"):
    primary_files_pattern = os.path.join(primary_data_dir, f"*{primary_suffix}")
    all_primary_file_paths = sorted(glob.glob(primary_files_pattern))
    base_primary_paths = []
    base_outline_paths = []
    for p_fpath in all_primary_file_paths:
        if "_aug" not in os.path.basename(p_fpath):
            base_filename = os.path.basename(p_fpath).replace(primary_suffix, "")
            outline_fpath = os.path.join(outline_dir, f"{base_filename}_outline.npy")
            if os.path.exists(outline_fpath):
                base_primary_paths.append(p_fpath)
                base_outline_paths.append(outline_fpath)
    return base_primary_paths, base_outline_paths

BASE_GRAYSCALE_PATHS, BASE_OUTLINE_PATHS = get_base_image_paths(
    GRAYSCALE_CATS_DIR_SAMPLING,
    OUTLINE_MASKS_DIR_SAMPLING,
    primary_suffix="_grayscale.npy"
)
if not BASE_GRAYSCALE_PATHS:
    print("Warning: No base grayscale images found. Example selection might fail.")

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
    """Converts a NumPy array (0-1 float or 0-255 uint8) to QPixmap."""
    if np_array.ndim == 3 and np_array.shape[2] == 1:
        np_array = np_array.squeeze()
    
    if np_array.dtype == np.float32 or np_array.dtype == np.float64: # Assume 0-1 range
        np_array = (np_array * 255).astype(np.uint8)
    elif np_array.dtype != np.uint8:
        np_array = np_array.astype(np.uint8)

    if is_mask: # Mask: 0 for object (black), 1 for background (white) from model. Invert for display.
                # Or if it's 0 object, 255 background from user drawing, it's fine.
                # The model output outline is 0=object, 1=bg. We display with gray_r, so it looks right.
                # For QLabel, QImage.Format_Grayscale8 expects 0=black, 255=white.
                # If our mask is 0=object, 1=bg -> convert to 0=object, 255=bg
        pass # Already (0 or 255) from model input mask generation.

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
    # (This function can remain largely the same as your last version, using Matplotlib for display)
    print("\n--- Starting Dataset Outline Prediction Mode ---")
    if not BASE_GRAYSCALE_PATHS or not BASE_OUTLINE_PATHS:
        print("Error: No base images found in dataset. Cannot proceed.")
        return

    num_tests = min(NUM_DATASET_TEST_ITERATIONS, len(BASE_OUTLINE_PATHS))
    if len(BASE_OUTLINE_PATHS) < NUM_DATASET_TEST_ITERATIONS:
        print(f"Warning: Requested {NUM_DATASET_TEST_ITERATIONS} tests, but only {len(BASE_OUTLINE_PATHS)} base outlines available.")

    selected_outline_indices = random.sample(range(len(BASE_OUTLINE_PATHS)), num_tests)

    for i, outline_idx in enumerate(selected_outline_indices):
        print(f"\n--- Test Iteration {i+1}/{num_tests} ---")
        input_outline_path = BASE_OUTLINE_PATHS[outline_idx]
        input_outline_img_list = load_images_from_paths([input_outline_path], IMG_SHAPE)
        if not input_outline_img_list: continue
        input_outline_img = input_outline_img_list[0]

        example_paths_to_load = []
        available_example_indices = list(range(len(BASE_GRAYSCALE_PATHS)))
        num_examples_to_sample = min(NUM_EXAMPLES_CONDITION, len(available_example_indices))

        selected_example_indices = random.sample(available_example_indices, num_examples_to_sample)
        for ex_idx in selected_example_indices:
            example_paths_to_load.append(BASE_GRAYSCALE_PATHS[ex_idx])
        
        example_images = load_images_from_paths(example_paths_to_load, IMG_SHAPE)
        if len(example_images) != num_examples_to_sample: continue
            
        print(f"Using outline: {os.path.basename(input_outline_path)}")
        generated_img = predict_single(trained_model, example_images, input_outline_img)
        if generated_img is not None:
            # Using Matplotlib for this mode
            plt.figure(figsize=(18, 4))
            for j, ex_img in enumerate(example_images):
                plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, j + 1); plt.imshow(ex_img.squeeze(), cmap='gray', vmin=0, vmax=1); plt.title(f"Ex {j+1}"); plt.axis('off')
            plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 1); plt.imshow(input_outline_img.squeeze(), cmap='gray_r'); plt.title("Input Outline"); plt.axis('off')
            plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 2); plt.imshow(generated_img.squeeze(), cmap='gray', vmin=0, vmax=1); plt.title("Generated"); plt.axis('off')
            plt.suptitle(f"Dataset Prediction {i+1}"); plt.tight_layout(rect=[0,0,1,0.95]); plt.show()
    print("\n--- Dataset Outline Prediction Mode Finished ---")


# --- PyQt5 Drawing Canvas Mode ---
class DrawingScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_path = None
        self.points = []
        self.is_drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.current_path = QPainterPath()
            self.current_path.moveTo(event.scenePos())
            self.points = [event.scenePos()] # Start new list of points
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing and (event.buttons() & Qt.LeftButton):
            self.current_path.lineTo(event.scenePos())
            self.points.append(event.scenePos())
            self.update_path_item() # Update incrementally
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            # Finalize path; it's already added/updated
            super().mouseReleaseEvent(event)
            
    def update_path_item(self):
        # Clear previous path items if any for incremental update
        for item in self.items():
            if isinstance(item, QGraphicsPathItem): # Assuming only one path for simplicity
                self.removeItem(item)
        
        if self.current_path:
            pen = QPen(Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            self.addPath(self.current_path, pen)

    def clear_scene(self):
        self.points = []
        if self.current_path:
            self.current_path = None # Reset path
        for item in self.items(): # Clear all items
            self.removeItem(item)
        self.update() # Force redraw

    def get_drawn_mask(self, target_dim):
        """Renders the scene to a QImage, then to NumPy array, processes, and scales."""
        if not self.points:
            return None

        # Render scene to QImage
        image = QImage(DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.white) # White background
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        # self.render(painter) # This renders all items, including potentially old ones if not cleared
        
        # Draw the current path manually to ensure it's the latest
        if self.current_path:
            pen = QPen(Qt.black, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(self.current_path)
        
        # If path is open and has points, attempt to close it by drawing line from last to first
        if len(self.points) > 2:
            start_qpoint = self.points[0]
            end_qpoint = self.points[-1]
            if (start_qpoint.x() - end_qpoint.x())**2 + (start_qpoint.y() - end_qpoint.y())**2 > 10**2: # Heuristic
                path_to_close = QPainterPath()
                path_to_close.moveTo(end_qpoint)
                path_to_close.lineTo(start_qpoint)
                painter.drawPath(path_to_close)

        painter.end()

        # Convert QImage to NumPy (OpenCV format)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4) # BGRA
        
        # Convert to Grayscale and then binary mask (object is 0, background is 255)
        gray_arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY) # Drawn lines are black (0)
        
        # To fill: find contours on inverted image (drawn lines white)
        inverted_gray = cv2.bitwise_not(gray_arr) # Now drawn lines are white (255)
        contours, _ = cv2.findContours(inverted_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filled_mask_512 = np.ones((DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE), dtype=np.uint8) * 255 # White background
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled_mask_512, [largest_contour], -1, 0, thickness=cv2.FILLED) # Fill object with black (0)
        
        # Scale to model input size (object 0, background 255)
        scaled_mask_cv = cv2.resize(filled_mask_512, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)
        
        # Normalize for model (object 0, background 1)
        model_input_mask_np = (scaled_mask_cv.astype(np.float32) / 255.0).reshape((target_dim, target_dim, 1))
        
        return model_input_mask_np, scaled_mask_cv # Return both for display and model

class DrawingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.example_model_inputs = []
        self.initUI()
        self.load_example_images()

    def initUI(self):
        self.setWindowTitle('Interactive Cat Generation')
        self.setGeometry(100, 100, DRAWING_CANVAS_SIZE + IMG_DIM*2 + 100, DRAWING_CANVAS_SIZE + 150) # Adjusted window size

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Top: Example Images
        self.examples_layout = QHBoxLayout()
        self.example_labels = [QLabel(f"Ex {i+1}") for i in range(NUM_EXAMPLES_CONDITION)]
        for label in self.example_labels:
            label.setFixedSize(PREVIEW_IMAGE_SIZE, PREVIEW_IMAGE_SIZE)
            label.setStyleSheet("border: 1px solid gray;")
            label.setAlignment(Qt.AlignCenter)
            self.examples_layout.addWidget(label)
        main_layout.addLayout(self.examples_layout)

        # Middle: Drawing Canvas and Result Displays
        drawing_results_layout = QHBoxLayout()
        main_layout.addLayout(drawing_results_layout)

        # Drawing Area
        self.scene = DrawingScene()
        self.scene.setSceneRect(0, 0, DRAWING_CANVAS_SIZE -2, DRAWING_CANVAS_SIZE -2) # Slightly smaller to see border
        self.view = QGraphicsView(self.scene)
        self.view.setFixedSize(DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE)
        self.view.setBackgroundBrush(Qt.white)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        drawing_results_layout.addWidget(self.view)

        # Processed Input Mask Display
        self.input_mask_label = QLabel("Input Mask")
        self.input_mask_label.setFixedSize(IMG_DIM, IMG_DIM)
        self.input_mask_label.setStyleSheet("border: 1px solid blue;")
        self.input_mask_label.setAlignment(Qt.AlignCenter)
        drawing_results_layout.addWidget(self.input_mask_label)

        # Generated Output Display
        self.generated_image_label = QLabel("Generated Image")
        self.generated_image_label.setFixedSize(IMG_DIM, IMG_DIM)
        self.generated_image_label.setStyleSheet("border: 1px solid green;")
        self.generated_image_label.setAlignment(Qt.AlignCenter)
        drawing_results_layout.addWidget(self.generated_image_label)

        # Bottom: Buttons
        buttons_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate (g)")
        self.generate_button.clicked.connect(self.on_generate)
        self.clear_button = QPushButton("Clear (c)")
        self.clear_button.clicked.connect(self.on_clear)
        buttons_layout.addWidget(self.generate_button)
        buttons_layout.addWidget(self.clear_button)
        main_layout.addLayout(buttons_layout)
        
        self.show()

    def load_example_images(self):
        if not BASE_GRAYSCALE_PATHS: return
        
        num_to_sample = min(NUM_EXAMPLES_CONDITION, len(BASE_GRAYSCALE_PATHS))
        selected_example_paths_indices = random.sample(range(len(BASE_GRAYSCALE_PATHS)), num_to_sample)
        
        selected_example_paths = [BASE_GRAYSCALE_PATHS[idx] for idx in selected_example_paths_indices]
        
        self.example_model_inputs = load_images_from_paths(selected_example_paths, IMG_SHAPE)
        
        for i in range(NUM_EXAMPLES_CONDITION):
            if i < len(self.example_model_inputs):
                pixmap = np_to_qpixmap(self.example_model_inputs[i], PREVIEW_IMAGE_SIZE)
                self.example_labels[i].setPixmap(pixmap)
            else: # Clear label if not enough examples
                self.example_labels[i].clear()
                self.example_labels[i].setText(f"Ex {i+1} N/A")


    def on_generate(self):
        print("Generate button clicked.")
        if not self.example_model_inputs or len(self.example_model_inputs) != NUM_EXAMPLES_CONDITION:
            print("Not enough example images loaded.")
            self.generated_image_label.setText("Load Examples!")
            return

        mask_data = self.scene.get_drawn_mask(IMG_DIM)
        if mask_data is None:
            print("No outline drawn.")
            self.input_mask_label.setText("Draw Outline!")
            return
        
        model_input_mask_np, display_mask_cv = mask_data # model_input_mask_np is 0-1, display_mask_cv is 0/255

        # Display the processed input mask
        pixmap_input_mask = np_to_qpixmap(display_mask_cv, IMG_DIM, is_mask=True) # display_mask_cv (0=obj, 255=bg)
        self.input_mask_label.setPixmap(pixmap_input_mask)

        generated_img_np = predict_single(trained_model, self.example_model_inputs, model_input_mask_np)
        if generated_img_np is not None:
            pixmap_generated = np_to_qpixmap(generated_img_np, IMG_DIM)
            self.generated_image_label.setPixmap(pixmap_generated)
            print("Generation complete.")
        else:
            self.generated_image_label.setText("Generation Failed")
            print("Generation failed.")

    def on_clear(self):
        print("Clear button clicked.")
        self.scene.clear_scene()
        self.input_mask_label.setText("Input Mask")
        self.input_mask_label.clear() # Clear pixmap
        self.generated_image_label.setText("Generated Image")
        self.generated_image_label.clear() # Clear pixmap

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_G:
            self.on_generate()
        elif event.key() == Qt.Key_C:
            self.on_clear()
        elif event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            self.close()


def qt_drawing_mode():
    app = QApplication(sys.argv)
    ex = DrawingWindow()
    sys.exit(app.exec_())

# --- Main Execution ---
def main():
    if USE_DATASET_OUTLINES and USE_DRAWING_CANVAS:
        print("Error: Both modes True. Choose one.")
        return
    if not USE_DATASET_OUTLINES and not USE_DRAWING_CANVAS:
        print("Error: No mode selected.")
        return

    if USE_DATASET_OUTLINES:
        dataset_prediction_mode()
    
    if USE_DRAWING_CANVAS:
        if not BASE_GRAYSCALE_PATHS:
            print("CRITICAL: No base grayscale images found for drawing mode examples. Exiting.")
            return
        qt_drawing_mode()

if __name__ == "__main__":
    main()