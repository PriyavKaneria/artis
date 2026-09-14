# src/predict.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob
import cv2 

# --- PyQt5 Imports ---
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QGraphicsView, QGraphicsScene, QGraphicsPathItem)
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap, QPainterPath
from PyQt5.QtCore import Qt, QPointF

# --- Import custom layers and model builder ---
from model_def import (FiLMLayer, PatchEmbed, StyleEncoder, SpatialPlanner, 
                       AdaptiveBottleneck, AdaptiveGenerator, DistanceWeightedSoftSpatialLoss)

# --- Configuration ---
IMG_DIM = 128
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
MODEL_PATH = "trained_models/hybrid_generator_v1.keras" # Ensure this matches saved model

OUTPUT_DIR_BASE_PROCESSED = "dataset/processed/grayscale_augmented_cat_v3/"
GRAYSCALE_CATS_DIR_SAMPLING = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "grayscale_cats/")
OUTLINE_MASKS_DIR_SAMPLING = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "outline_masks/")

USE_DATASET_OUTLINES = False
USE_DRAWING_CANVAS = True
NUM_DATASET_TEST_ITERATIONS = 3
DRAWING_CANVAS_SIZE = 512
PREVIEW_IMAGE_SIZE = 64

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}.")
try:
    custom_objects_dict = {
        'FiLMLayer': FiLMLayer, 'PatchEmbed': PatchEmbed, 'StyleEncoder': StyleEncoder,
        'SpatialPlanner': SpatialPlanner, 'AdaptiveBottleneck': AdaptiveBottleneck,
        'AdaptiveGenerator': AdaptiveGenerator, 'DistanceWeightedSoftSpatialLoss': DistanceWeightedSoftSpatialLoss
    }
    trained_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects_dict)
    print("Trained Hybrid model loaded successfully.")
except Exception as e:
    print(f"Error loading Hybrid model: {e}"); exit()

# --- Helper Functions (mostly same as previous predict.py) ---
def get_base_grayscale_paths(grayscale_dir): # Only for examples
    gs_pattern = os.path.join(grayscale_dir, "*_grayscale.npy")
    all_gs_paths = sorted(glob.glob(gs_pattern))
    return [p for p in all_gs_paths if "_aug" not in p.split('/')[-1] and "_approx" not in p.split('/')[-1]]

BASE_GRAYSCALE_PATHS_FOR_EXAMPLES = get_base_grayscale_paths(GRAYSCALE_CATS_DIR_SAMPLING)
ALL_AVAILABLE_OUTLINE_PATHS = sorted(glob.glob(os.path.join(OUTLINE_MASKS_DIR_SAMPLING, "*_outline.npy")))

if not BASE_GRAYSCALE_PATHS_FOR_EXAMPLES: print("Warning: No base grayscales for examples.")
if not ALL_AVAILABLE_OUTLINE_PATHS: print("Warning: No outlines found for dataset mode.")


def load_images_from_paths(paths_list, img_shape_tuple):
    images = []
    for path in paths_list:
        try: images.append(np.load(path).reshape(img_shape_tuple))
        except Exception as e: print(f"Warn: Load fail {path}: {e}")
    return images

def np_to_qpixmap(np_array, target_size=None, is_mask=False):
    # (Same np_to_qpixmap as previous predict.py)
    if np_array.ndim == 3 and np_array.shape[2] == 1: np_array = np_array.squeeze()
    if np_array.dtype == np.float32 or np_array.dtype == np.float64: np_array = (np_array * 255).astype(np.uint8)
    elif np_array.dtype != np.uint8: np_array = np_array.astype(np.uint8)
    h, w = np_array.shape
    q_img = QImage(np_array.data, w, h, w, QImage.Format_Grayscale8)
    pixmap = QPixmap.fromImage(q_img)
    if target_size: pixmap = pixmap.scaled(target_size, target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return pixmap

def predict_single(model, example_imgs_list, input_outline_mask):
    if len(example_imgs_list) != NUM_EXAMPLES_CONDITION: return None
    ex_tensors = [np.expand_dims(img, axis=0) for img in example_imgs_list]
    ol_tensor = np.expand_dims(input_outline_mask, axis=0)
    return model.predict(ex_tensors + [ol_tensor])[0]


# --- Dataset Outline Mode ---
def dataset_prediction_mode():
    # (Same as previous predict.py, just ensure it uses BASE_GRAYSCALE_PATHS_FOR_EXAMPLES for examples
    # and samples from ALL_AVAILABLE_OUTLINE_PATHS for the input_outline_path)
    print("\n--- Dataset Outline Prediction Mode ---")
    if not BASE_GRAYSCALE_PATHS_FOR_EXAMPLES or not ALL_AVAILABLE_OUTLINE_PATHS: print("Missing data for dataset mode."); return
    num_tests = min(NUM_DATASET_TEST_ITERATIONS, len(ALL_AVAILABLE_OUTLINE_PATHS))
    selected_outline_paths = random.sample(ALL_AVAILABLE_OUTLINE_PATHS, num_tests)

    for i, ol_path in enumerate(selected_outline_paths):
        print(f"\nTest {i+1}/{num_tests}, Outline: {os.path.basename(ol_path)}")
        ol_img_list = load_images_from_paths([ol_path], IMG_SHAPE)
        if not ol_img_list: continue
        ol_img = ol_img_list[0]

        ex_paths = random.sample(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES, min(NUM_EXAMPLES_CONDITION, len(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES)))
        ex_imgs = load_images_from_paths(ex_paths, IMG_SHAPE)
        if len(ex_imgs) != NUM_EXAMPLES_CONDITION: print(f"Not enough examples for {os.path.basename(ol_path)}"); continue
        
        gen_img = predict_single(trained_model, ex_imgs, ol_img)
        if gen_img is not None:
            plt.figure(figsize=(3 * (len(ex_imgs) + 2), 3) )
            for j, exi in enumerate(ex_imgs): plt.subplot(1,len(ex_imgs)+2,j+1); plt.imshow(exi.squeeze(),cmap='gray',vmin=0,vmax=1); plt.title(f"Ex{j+1}"); plt.axis('off')
            plt.subplot(1,len(ex_imgs)+2,len(ex_imgs)+1); plt.imshow(ol_img.squeeze(),cmap='gray_r'); plt.title("Input OL"); plt.axis('off')
            plt.subplot(1,len(ex_imgs)+2,len(ex_imgs)+2); plt.imshow(gen_img.squeeze(),cmap='gray',vmin=0,vmax=1); plt.title("Generated"); plt.axis('off')
            plt.suptitle(f"Dataset Pred {i+1}"); plt.tight_layout(rect=[0,0,1,0.93]); plt.show()
    print("--- Dataset Mode Finished ---")

# --- PyQt5 Drawing Canvas Mode ---
# (The DrawingScene and DrawingWindow classes can remain largely the same as your last version)
class DrawingScene(QGraphicsScene): # (Same as your previous PyQt DrawingScene)
    def __init__(self, parent=None): super().__init__(parent); self.current_path = None; self.points = []; self.is_drawing = False
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.is_drawing = True
        if not self.current_path or not self.is_drawing: self.current_path = QPainterPath(); self.current_path.moveTo(event.scenePos()); self.points = [event.scenePos()]
        else: self.current_path.lineTo(event.scenePos()); self.points.append(event.scenePos())
        self.update_path_item(); super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self.is_drawing and (event.buttons() & Qt.LeftButton):
            if self.current_path: self.current_path.lineTo(event.scenePos()); self.points.append(event.scenePos()); self.update_path_item()
            super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing: self.is_drawing = False; super().mouseReleaseEvent(event)
    def update_path_item(self):
        items_to_remove = [item for item in self.items() if isinstance(item, QGraphicsPathItem)]; [self.removeItem(item) for item in items_to_remove]
        if self.current_path: self.addPath(self.current_path, QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    def clear_scene(self): self.points = []; self.current_path = None; [self.removeItem(item) for item in self.items() if isinstance(item, QGraphicsPathItem)]; self.update()
    def get_drawn_mask(self, target_dim): # (Same as your previous version)
        if not self.points and not self.current_path: return None
        image = QImage(DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE, QImage.Format_ARGB32_Premultiplied); image.fill(Qt.white)
        painter = QPainter(image); painter.setRenderHint(QPainter.Antialiasing)
        if self.current_path: painter.setPen(QPen(Qt.black, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)); painter.drawPath(self.current_path)
        if self.points and len(self.points) > 2:
            s, e = self.points[0], self.points[-1]
            if (s.x()-e.x())**2 + (s.y()-e.y())**2 > 15**2: cp=QPainterPath(); cp.moveTo(e); cp.lineTo(s); painter.drawPath(cp)
        painter.end(); ptr = image.bits(); ptr.setsize(image.byteCount()); arr = np.array(ptr).reshape(image.height(),image.width(),4)
        gray_arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY); inverted_gray = cv2.bitwise_not(gray_arr)
        contours, _ = cv2.findContours(inverted_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask_512 = np.ones((DRAWING_CANVAS_SIZE, DRAWING_CANVAS_SIZE), dtype=np.uint8)*255
        if contours: cv2.drawContours(filled_mask_512, [max(contours, key=cv2.contourArea)], -1, 0, thickness=cv2.FILLED)
        scaled_mask_cv = cv2.resize(filled_mask_512, (target_dim, target_dim), interpolation=cv2.INTER_NEAREST)
        return (scaled_mask_cv.astype(np.float32)/255.0).reshape((target_dim,target_dim,1)), scaled_mask_cv

class DrawingWindow(QWidget): # (Same as your previous PyQt DrawingWindow)
    def __init__(self): super().__init__(); self.example_model_inputs = []; self.initUI(); self.load_example_images()
    def initUI(self):
        self.setWindowTitle('Interactive Cat Gen (Hybrid Model)'); self.setGeometry(100,100,DRAWING_CANVAS_SIZE+IMG_DIM*2+100,DRAWING_CANVAS_SIZE+150)
        ml=QVBoxLayout();self.setLayout(ml);el=QHBoxLayout();self.ex_labels=[QLabel(f"Ex{i+1}")for i in range(NUM_EXAMPLES_CONDITION)]
        for lbl in self.ex_labels:lbl.setFixedSize(PREVIEW_IMAGE_SIZE,PREVIEW_IMAGE_SIZE);lbl.setStyleSheet("border:1px solid gray;");lbl.setAlignment(Qt.AlignCenter);el.addWidget(lbl)
        ml.addLayout(el);drl=QHBoxLayout();ml.addLayout(drl);self.scene=DrawingScene();self.scene.setSceneRect(0,0,DRAWING_CANVAS_SIZE-2,DRAWING_CANVAS_SIZE-2)
        self.view=QGraphicsView(self.scene);self.view.setFixedSize(DRAWING_CANVAS_SIZE,DRAWING_CANVAS_SIZE);self.view.setBackgroundBrush(Qt.white);self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);drl.addWidget(self.view)
        self.im_lbl=QLabel("InputMask");self.im_lbl.setFixedSize(IMG_DIM,IMG_DIM);self.im_lbl.setStyleSheet("border:1px solid blue;");self.im_lbl.setAlignment(Qt.AlignCenter);drl.addWidget(self.im_lbl)
        self.gi_lbl=QLabel("GenImage");self.gi_lbl.setFixedSize(IMG_DIM,IMG_DIM);self.gi_lbl.setStyleSheet("border:1px solid green;");self.gi_lbl.setAlignment(Qt.AlignCenter);drl.addWidget(self.gi_lbl)
        bl=QHBoxLayout();self.gen_btn=QPushButton("Gen(g)");self.gen_btn.clicked.connect(self.on_gen);self.clr_btn=QPushButton("Clr(c)");self.clr_btn.clicked.connect(self.on_clr)
        bl.addWidget(self.gen_btn);bl.addWidget(self.clr_btn);ml.addLayout(bl);self.show()
    def load_example_images(self):
        if not BASE_GRAYSCALE_PATHS_FOR_EXAMPLES:return
        num_samp=min(NUM_EXAMPLES_CONDITION,len(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES));paths=random.sample(BASE_GRAYSCALE_PATHS_FOR_EXAMPLES,num_samp)
        self.example_model_inputs=load_images_from_paths(paths,IMG_SHAPE)
        for i in range(NUM_EXAMPLES_CONDITION):
            if i<len(self.example_model_inputs):self.ex_labels[i].setPixmap(np_to_qpixmap(self.example_model_inputs[i],PREVIEW_IMAGE_SIZE))
            else:self.ex_labels[i].clear();self.ex_labels[i].setText(f"Ex{i+1} N/A")
    def on_gen(self):
        if not self.example_model_inputs or len(self.example_model_inputs)!=NUM_EXAMPLES_CONDITION:print("No ex imgs");return
        mdata=self.scene.get_drawn_mask(IMG_DIM);
        if mdata is None:print("No outline");return
        model_mask_np,disp_mask_cv=mdata;self.im_lbl.setPixmap(np_to_qpixmap(disp_mask_cv,IMG_DIM,True))
        gen_img=predict_single(trained_model,self.example_model_inputs,model_mask_np)
        if gen_img is not None:self.gi_lbl.setPixmap(np_to_qpixmap(gen_img,IMG_DIM));print("Generated.")
        else:self.gi_lbl.setText("Gen Failed");print("Gen fail.")
    def on_clr(self):self.scene.clear_scene();self.im_lbl.setText("InputMask");self.im_lbl.clear();self.gi_lbl.setText("GenImage");self.gi_lbl.clear()
    def keyPressEvent(self,e):
        if e.key()==Qt.Key_G:self.on_gen()
        elif e.key()==Qt.Key_C:self.on_clr()
        elif e.key()==Qt.Key_Q or e.key()==Qt.Key_Escape:self.close()

def qt_drawing_mode(): app=QApplication.instance() or QApplication(sys.argv); ex=DrawingWindow(); sys.exit(app.exec_())
def main():
    if USE_DATASET_OUTLINES: dataset_prediction_mode()
    if USE_DRAWING_CANVAS: qt_drawing_mode()
if __name__=="__main__": main()