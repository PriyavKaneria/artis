PROJECT SUMMARY & CURRENT FOCUS

**Overall Vision (Long-Term):**
To develop a system with a supervisor/operator model (e.g., BERT-based) that intelligently selects and deploys specialized "tiny" image generation models. Each tiny model would be an expert at generating a specific kind of object within a user-defined outline.

**Current Achievable Step (Proof-of-Concept):**
Train a single, very small, non-diffusion model to generate B&W pixel art of a specific object (e.g., a cat) within a given outline mask. The generation is conditioned on 5 example images that provide stylistic and detail cues.

**Core Task Breakdown (Current Focus):**
1.  **Input to the Model:**
    *   Five (5) "Edge Detail Images": 128x128 B&W images where the object's internal edges and details are black (0) on a white (1) background. These are derived from original cat PNGs using Canny edge detection.
    *   One (1) "Outline Mask Image": A 128x128 B&W image where the filled silhouette of the target object is black (0) on a white (1) background. This mask defines the area to be filled.
2.  **Output from the Model:**
    *   One (1) 128x128 B&W pixel art image. The model attempts to reconstruct the "Outline Mask Image" (which acts as the ground truth during training), but its internal structure/texture should be influenced by the 5 "Edge Detail Images."
3.  **Image Type:** Black & White pixel art (pixel values are 0.0 or 1.0).
4.  **Resolution:** 128x128 pixels for all images.
5.  **Model Constraints:**
    *   Parameter Count: Less than 2 million parameters.
    *   Inference Speed: Prioritized.
6.  **Data Augmentation:** Currently not implemented, but strongly recommended for future robustness.
7.  **Model Architecture:** A Conditional U-Net architecture. An example encoder processes the 5 edge detail images to produce a conditioning vector, which is then fed into the U-Net along with the outline mask.

---

PROJECT DIRECTORY STRUCTURE

cat_pixel_generator/
├── original_cat_pngs/      # Source PNG images of cats (with alpha channels)
├── processed_data/
│   ├── outline_masks/      # Stores 128x128 B&W filled outline masks (.npy)
│   └── edge_details/       # Stores 128x128 B&W edge detail images (.npy)
├── src/
│   ├── prepare_data.py     # Script to process PNGs into outline masks and edge details
│   ├── model_def.py        # Defines the neural network architecture (TensorFlow/Keras)
│   ├── train.py            # Main training script
│   ├── predict.py          # Script for inference and visualization
│   └── utils.py            # (Optional) Helper functions
└── trained_models/         # Where final model weights will be saved (e.g., .keras files)

---

CODE FILES

**1. `src/prepare_data.py`**

*   **Purpose:** Converts original RGBA PNG images of cats into two types of 128x128 B&W .npy files:
    *   `outline_masks`: Filled silhouettes of the cat (cat=0, background=1).
    *   `edge_details`: Canny-detected edges of the cat (edges=0, background=1), with pre-blurring and post-dilation options.
    Both are scaled to fit within the 128x128 dimensions while maintaining aspect ratio and centered.

```python
# src/prepare_data.py
import cv2
import numpy as np
from PIL import Image
import os
import glob

TARGET_SIZE = (128, 128)
OUTPUT_DIR_BASE = "../processed_data"
OUTPUT_DIR_OUTLINES = os.path.join(OUTPUT_DIR_BASE, "outline_masks")
OUTPUT_DIR_EDGEDETAILS = os.path.join(OUTPUT_DIR_BASE, "edge_details")
INPUT_DIR_PNGS = "../original_cat_pngs"

# --- Parameters for Edge Enhancement ---
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5) 
GAUSSIAN_BLUR_SIGMAX = 0 
CANNY_THRESHOLD1 = 30
CANNY_THRESHOLD2 = 100
DILATION_KERNEL_SIZE = (2, 2)
DILATION_ITERATIONS = 1
# --- End of Edge Enhancement Parameters ---

SCALING_PADDING = 4

def create_filled_outline_mask(precise_cat_roi_mask_orig_res, target_size):
    if precise_cat_roi_mask_orig_res is None or precise_cat_roi_mask_orig_res.size == 0:
        return None
    outline_image_final = np.ones(target_size, dtype=np.float32)
    roi_h, roi_w = precise_cat_roi_mask_orig_res.shape
    if roi_w == 0 or roi_h == 0: return None
    scale = min((target_size[0] - SCALING_PADDING) / roi_w, (target_size[1] - SCALING_PADDING) / roi_h)
    if scale <= 0: return None
    new_w, new_h = max(1, int(roi_w * scale)), max(1, int(roi_h * scale))
    resized_cat_roi_mask = cv2.resize(precise_cat_roi_mask_orig_res, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    start_x = (target_size[0] - new_w) // 2
    start_y = (target_size[1] - new_h) // 2
    outline_image_final[start_y:start_y+new_h, start_x:start_x+new_w][resized_cat_roi_mask == 255] = 0.0
    return outline_image_final

def create_bnw_edge_detail_image(cat_color_roi_orig_res, precise_cat_roi_mask_orig_res, target_size):
    if cat_color_roi_orig_res is None or precise_cat_roi_mask_orig_res is None:
        return None
    gray_cat_color_roi = cv2.cvtColor(cat_color_roi_orig_res, cv2.COLOR_RGB2GRAY)
    masked_gray_for_canny_input = cv2.bitwise_and(gray_cat_color_roi, gray_cat_color_roi, mask=precise_cat_roi_mask_orig_res)
    blurred_img = masked_gray_for_canny_input
    if GAUSSIAN_BLUR_KERNEL_SIZE and GAUSSIAN_BLUR_KERNEL_SIZE[0] > 0 and GAUSSIAN_BLUR_KERNEL_SIZE[1] > 0:
        blurred_img = cv2.GaussianBlur(masked_gray_for_canny_input, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMAX)
    edges_canny = cv2.Canny(blurred_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
    dilated_edges = edges_canny
    if DILATION_ITERATIONS > 0 and DILATION_KERNEL_SIZE and DILATION_KERNEL_SIZE[0] > 0 and DILATION_KERNEL_SIZE[1] > 0:
        dilation_kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
        dilated_edges = cv2.dilate(edges_canny, dilation_kernel, iterations=DILATION_ITERATIONS)
    edge_detail_image_final = np.ones(target_size, dtype=np.float32)
    roi_h, roi_w = precise_cat_roi_mask_orig_res.shape
    if roi_w == 0 or roi_h == 0: return None
    scale = min((target_size[0] - SCALING_PADDING) / roi_w, (target_size[1] - SCALING_PADDING) / roi_h)
    if scale <= 0: return None
    new_w, new_h = max(1, int(roi_w * scale)), max(1, int(roi_h * scale))
    resized_edges = cv2.resize(dilated_edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    start_x = (target_size[0] - new_w) // 2
    start_y = (target_size[1] - new_h) // 2
    edge_detail_image_final[start_y:start_y+new_h, start_x:start_x+new_w][resized_edges == 255] = 0.0
    return edge_detail_image_final

def main():
    os.makedirs(OUTPUT_DIR_OUTLINES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_EDGEDETAILS, exist_ok=True)
    image_paths = glob.glob(os.path.join(INPUT_DIR_PNGS, "*.png"))
    if not image_paths:
        print(f"No PNG files found in {INPUT_DIR_PNGS}.")
        return
    print(f"Found {len(image_paths)} PNG files. Starting processing to {TARGET_SIZE}...")
    print(f"Using Blur Kernel: {GAUSSIAN_BLUR_KERNEL_SIZE}, Canny Thresh: ({CANNY_THRESHOLD1},{CANNY_THRESHOLD2}), Dilation Kernel: {DILATION_KERNEL_SIZE} Iter: {DILATION_ITERATIONS}")
    processed_count = 0
    for i, image_path in enumerate(image_paths):
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing ({i+1}/{len(image_paths)}): {filename_base}")
        try:
            img_pil_orig = Image.open(image_path).convert("RGBA")
            img_rgba_orig = np.array(img_pil_orig)
            if img_rgba_orig.shape[2] != 4:
                print(f"  Skipping {filename_base}: Not an RGBA image.")
                continue
            alpha_channel_full = img_rgba_orig[:, :, 3]
            _, binary_mask_from_alpha_full_res = cv2.threshold(alpha_channel_full, 127, 255, cv2.THRESH_BINARY)
            if np.sum(binary_mask_from_alpha_full_res) == 0:
                print(f"  Skipping {filename_base}: No foreground in alpha channel.")
                continue
            contours, _ = cv2.findContours(binary_mask_from_alpha_full_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"  Skipping {filename_base}: No contours found from alpha mask.")
                continue
            main_contour = max(contours, key=cv2.contourArea)
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(main_contour)
            precise_cat_roi_mask_orig_res = np.zeros((h_bbox, w_bbox), dtype=np.uint8)
            shifted_contour = main_contour - [x_bbox, y_bbox]
            cv2.drawContours(precise_cat_roi_mask_orig_res, [shifted_contour], -1, color=255, thickness=cv2.FILLED)
            cat_color_roi_orig_res = img_rgba_orig[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox, :3]
            outline_mask_img = create_filled_outline_mask(precise_cat_roi_mask_orig_res, TARGET_SIZE)
            if outline_mask_img is not None:
                save_path_outline = os.path.join(OUTPUT_DIR_OUTLINES, f"{filename_base}_outline.npy")
                np.save(save_path_outline, outline_mask_img)
                # cv2.imwrite(os.path.join(OUTPUT_DIR_OUTLINES, f"{filename_base}_outline_preview.png"), (outline_mask_img * 255).astype(np.uint8))
            else:
                print(f"  Failed to create outline mask for {filename_base}. Skipping this image.")
                continue
            edge_detail_img = create_bnw_edge_detail_image(cat_color_roi_orig_res, precise_cat_roi_mask_orig_res, TARGET_SIZE)
            if edge_detail_img is not None:
                save_path_edge = os.path.join(OUTPUT_DIR_EDGEDETAILS, f"{filename_base}_edgedetail.npy")
                np.save(save_path_edge, edge_detail_img)
                cv2.imwrite(os.path.join(OUTPUT_DIR_EDGEDETAILS, f"{filename_base}_edgedetail_preview.png"), (edge_detail_img * 255).astype(np.uint8))
            else:
                print(f"  Failed to create edge detail image for {filename_base}. Skipping this image.")
                if os.path.exists(save_path_outline): os.remove(save_path_outline)
                continue
            processed_count += 1
        except Exception as e:
            print(f"  Unhandled error processing {filename_base}: {e}")
            continue
    print(f"\nFinished processing. Successfully created paired files for {processed_count} images.")
    print(f"Outline masks saved in: {OUTPUT_DIR_OUTLINES}")
    print(f"Edge detail images saved in: {OUTPUT_DIR_EDGEDETAILS}")

if __name__ == "__main__":
    main()
```

**2. `src/model_def.py`**

*   **Purpose:** Defines the TensorFlow/Keras model architecture, including:
    *   `build_example_encoder`: A CNN to process each of the 5 edge detail input images into a latent vector.
    *   `build_generator`: A U-Net style generator that takes the outline mask and the averaged latent vector (from example encoder) as input to produce the final image.
    *   `build_combined_model`: Assembles the example encoder and generator into a single trainable model.
    Input image shape is (128, 128, 1).

```python
# src/model_def.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    GlobalAveragePooling2D, Dense, Reshape, Add, Activation, BatchNormalization, Average
)
from tensorflow.keras.models import Model

def build_example_encoder(input_shape=(128, 128, 1), latent_dim=64, name="example_encoder"):
    img_input = Input(shape=input_shape, name=f"{name}_input")
    x = Conv2D(16, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv1")(img_input) # 64x64
    x = BatchNormalization(name=f"{name}_bn1")(x); x = Activation("relu", name=f"{name}_relu1")(x)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv2")(x) # 32x32
    x = BatchNormalization(name=f"{name}_bn2")(x); x = Activation("relu", name=f"{name}_relu2")(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv3")(x) # 16x16
    x = BatchNormalization(name=f"{name}_bn3")(x); x = Activation("relu", name=f"{name}_relu3")(x)
    x = Conv2D(latent_dim, (3, 3), strides=(2, 2), padding="same", name=f"{name}_conv4")(x) # 8x8
    x = BatchNormalization(name=f"{name}_bn4")(x); x = Activation("relu", name=f"{name}_relu4")(x)
    feature_vector = GlobalAveragePooling2D(name=f"{name}_gap")(x)
    encoder = Model(img_input, feature_vector, name=name)
    return encoder

def build_generator(outline_input_shape=(128, 128, 1), example_latent_dim=64, num_filters_base=16, name="generator"):
    outline_input = Input(shape=outline_input_shape, name=f"{name}_outline_input")
    condition_input = Input(shape=(example_latent_dim,), name=f"{name}_condition_input")

    # Encoder Path
    e1 = Conv2D(num_filters_base, (3,3), padding="same", name=f"{name}_e1_conv")(outline_input)
    e1 = BatchNormalization(name=f"{name}_e1_bn")(e1); e1 = Activation("relu", name=f"{name}_e1_relu")(e1) # 128x128
    p1 = MaxPooling2D((2,2), name=f"{name}_e1_pool")(e1) # 64x64
    e2 = Conv2D(num_filters_base*2, (3,3), padding="same", name=f"{name}_e2_conv")(p1)
    e2 = BatchNormalization(name=f"{name}_e2_bn")(e2); e2 = Activation("relu", name=f"{name}_e2_relu")(e2) # 64x64
    p2 = MaxPooling2D((2,2), name=f"{name}_e2_pool")(e2) # 32x32
    e3 = Conv2D(num_filters_base*4, (3,3), padding="same", name=f"{name}_e3_conv")(p2)
    e3 = BatchNormalization(name=f"{name}_e3_bn")(e3); e3 = Activation("relu", name=f"{name}_e3_relu")(e3) # 32x32
    p3 = MaxPooling2D((2,2), name=f"{name}_e3_pool")(e3) # 16x16
    e4 = Conv2D(num_filters_base*8, (3,3), padding="same", name=f"{name}_e4_conv")(p3)
    e4 = BatchNormalization(name=f"{name}_e4_bn")(e4); e4 = Activation("relu", name=f"{name}_e4_relu")(e4) # 16x16
    p4 = MaxPooling2D((2,2), name=f"{name}_e4_pool")(e4) # 8x8

    # Bottleneck
    b = Conv2D(num_filters_base*16, (3,3), padding="same", name=f"{name}_b_conv")(p4) # 8x8
    b = BatchNormalization(name=f"{name}_b_bn")(b); b = Activation("relu", name=f"{name}_b_relu")(b)
    cond_channels = b.shape[-1]; cond_spatial_dims = b.shape[1:3]
    projected_condition = Dense(cond_spatial_dims[0] * cond_spatial_dims[1] * cond_channels, activation='relu', name=f"{name}_cond_dense")(condition_input)
    reshaped_condition = Reshape((cond_spatial_dims[0], cond_spatial_dims[1], cond_channels), name=f"{name}_cond_reshape")(projected_condition)
    b_conditioned = Add(name=f"{name}_cond_add")([b, reshaped_condition])

    # Decoder Path
    d1 = UpSampling2D((2,2), name=f"{name}_d1_upsample")(b_conditioned) # 16x16
    d1 = concatenate([d1, e4], name=f"{name}_d1_concat")
    d1 = Conv2D(num_filters_base*8, (3,3), padding="same", name=f"{name}_d1_conv")(d1)
    d1 = BatchNormalization(name=f"{name}_d1_bn")(d1); d1 = Activation("relu", name=f"{name}_d1_relu")(d1)
    d2 = UpSampling2D((2,2), name=f"{name}_d2_upsample")(d1) # 32x32
    d2 = concatenate([d2, e3], name=f"{name}_d2_concat")
    d2 = Conv2D(num_filters_base*4, (3,3), padding="same", name=f"{name}_d2_conv")(d2)
    d2 = BatchNormalization(name=f"{name}_d2_bn")(d2); d2 = Activation("relu", name=f"{name}_d2_relu")(d2)
    d3 = UpSampling2D((2,2), name=f"{name}_d3_upsample")(d2) # 64x64
    d3 = concatenate([d3, e2], name=f"{name}_d3_concat")
    d3 = Conv2D(num_filters_base*2, (3,3), padding="same", name=f"{name}_d3_conv")(d3)
    d3 = BatchNormalization(name=f"{name}_d3_bn")(d3); d3 = Activation("relu", name=f"{name}_d3_relu")(d3)
    d4 = UpSampling2D((2,2), name=f"{name}_d4_upsample")(d3) # 128x128
    d4 = concatenate([d4, e1], name=f"{name}_d4_concat")
    d4 = Conv2D(num_filters_base, (3,3), padding="same", name=f"{name}_d4_conv")(d4)
    d4 = BatchNormalization(name=f"{name}_d4_bn")(d4); d4 = Activation("relu", name=f"{name}_d4_relu")(d4)
    output_image = Conv2D(1, (1,1), padding="same", activation="sigmoid", name=f"{name}_output_conv")(d4)
    generator_model = Model(inputs=[outline_input, condition_input], outputs=output_image, name=name)
    return generator_model

def build_combined_model(img_shape=(128, 128, 1), num_examples=5, example_latent_dim=64, gen_filters_base=16):
    example_img_inputs = [Input(shape=img_shape, name=f"example_img_input_{i}") for i in range(num_examples)]
    outline_input = Input(shape=img_shape, name="outline_main_input")
    example_encoder = build_example_encoder(input_shape=img_shape, latent_dim=example_latent_dim)
    example_features = [example_encoder(img_input) for img_input in example_img_inputs]
    if num_examples > 1:
        averaged_features = Average(name="average_example_features")(example_features)
    else:
        averaged_features = example_features[0]
    generator = build_generator(outline_input_shape=img_shape, example_latent_dim=example_latent_dim, num_filters_base=gen_filters_base)
    generated_output = generator([outline_input, averaged_features])
    combined_model = Model(inputs=example_img_inputs + [outline_input], outputs=generated_output, name="pixel_art_cat_generator")
    return combined_model, example_encoder, generator
```

**3. `src/train.py`**

*   **Purpose:** Handles the model training process.
    *   Loads the preprocessed "edge detail" and "outline mask" images.
    *   Defines a data generator that, for each training batch:
        *   Randomly samples 5 "edge detail" images as conditioning input.
        *   Randomly samples 1 "outline mask" image to serve as both the generator's outline input and the ground truth target for the loss function.
    *   Compiles the `combined_model` with an Adam optimizer and Binary Cross-Entropy loss.
    *   Trains the model and saves checkpoints.
    *   Plots training loss.

*   **Note:** The following `train.py` and `predict.py` will need slight modifications to load from the two separate data directories (`edge_details` and `outline_masks`). The code below is the previous version and needs this update. I will provide a diff for this separately if you confirm the data prep and model def are fine. For now, here's the version that loads from a single directory, assuming you'd merge or pair them manually before loading. **A more robust `train.py` would load corresponding pairs directly.**

```python
# src/train.py 
# !!! IMPORTANT: This version assumes edge_details are example inputs
# !!! and outline_masks are both the outline input AND the target output.
# !!! It needs modification to load from two separate directories and pair them correctly.
import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import glob # Added for loading paired data
from model_def import build_combined_model

# --- Configuration ---
IMG_DIM = 128
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
BATCH_SIZE = 8 # Adjusted for 128x128
EPOCHS = 200
LEARNING_RATE = 1e-4
EXAMPLE_LATENT_DIM = 64
GEN_FILTERS_BASE = 16

# Adjusted data directories
EDGE_DETAILS_DIR = "../processed_data/edge_details/"
OUTLINE_MASKS_DIR = "../processed_data/outline_masks/"
MODEL_SAVE_DIR = "../trained_models/"
MODEL_NAME = "cat_pixelart_generator_128.keras"

# --- Load Data (Modified to load pairs) ---
def load_paired_data(edge_dir, outline_dir):
    edge_files = sorted(glob.glob(os.path.join(edge_dir, "*.npy")))
    outline_files = sorted(glob.glob(os.path.join(outline_dir, "*.npy")))

    if not edge_files or not outline_files:
        raise FileNotFoundError(f"No .npy files found in {edge_dir} or {outline_dir}. Run prepare_data.py first.")
    
    # Basic pairing by filename (assuming prepare_data.py creates corresponding names)
    # e.g., cat1_edgedetail.npy and cat1_outline.npy
    edge_images_map = {os.path.basename(f).replace("_edgedetail.npy", ""): f for f in edge_files}
    outline_images_map = {os.path.basename(f).replace("_outline.npy", ""): f for f in outline_files}

    common_bases = sorted(list(set(edge_images_map.keys()) & set(outline_images_map.keys())))
    
    if not common_bases:
        raise ValueError("No matching base filenames found between edge details and outline masks.")

    loaded_edge_details = []
    loaded_outline_masks = [] # These will be both input outlines and targets

    for base in common_bases:
        edge_img = np.load(edge_images_map[base]).reshape(IMG_SHAPE)
        outline_img = np.load(outline_images_map[base]).reshape(IMG_SHAPE)
        loaded_edge_details.append(edge_img)
        loaded_outline_masks.append(outline_img) # This serves as target and input outline
        
    return np.array(loaded_edge_details), np.array(loaded_outline_masks)

all_edge_images_np, all_outline_masks_np = load_paired_data(EDGE_DETAILS_DIR, OUTLINE_MASKS_DIR)
print(f"Loaded {len(all_edge_images_np)} paired edge detail and outline mask images.")

if len(all_edge_images_np) < NUM_EXAMPLES_CONDITION + 1: # Need enough variety for sampling
    raise ValueError(f"Not enough images. Need at least {NUM_EXAMPLES_CONDITION + 1} distinct images for robust sampling.")

# --- Data Generator ---
def data_generator_fn(edge_images_dataset, outline_masks_dataset, batch_size, num_examples_cond):
    num_total_items = len(edge_images_dataset) # Assuming edge_images and outline_masks are paired and same length
    indices = np.arange(num_total_items)

    while True:
        np.random.shuffle(indices)
        
        for i in range(0, num_total_items - batch_size + 1, batch_size): # Iterate through data ensuring full batches
            batch_example_inputs_list = [[] for _ in range(num_examples_cond)]
            batch_outline_inputs_for_gen = [] # Outline mask fed to generator
            batch_target_outputs = []           # Outline mask as target

            for _ in range(batch_size): # For each item in the batch
                # Sample NUM_EXAMPLES_CONDITION distinct edge images for conditioning
                # These can be any edge images from the dataset
                example_indices = random.sample(range(num_total_items), num_examples_cond)
                sampled_example_edge_imgs = edge_images_dataset[example_indices]

                # Pick one outline_mask to be the target and the input outline for this instance
                # It's important that this index is NOT necessarily one of the example_indices
                # unless you want the model to learn reconstruction from its own style.
                # For generalization, pick a random outline mask.
                target_outline_idx = random.choice(indices) # Pick any index from the shuffled list for target
                
                current_outline_mask = outline_masks_dataset[target_outline_idx]
                
                for k in range(num_examples_cond):
                    batch_example_inputs_list[k].append(sampled_example_edge_imgs[k])
                
                batch_outline_inputs_for_gen.append(current_outline_mask) # This is the B&W pixel art shape
                batch_target_outputs.append(current_outline_mask)      # Target is to reconstruct this
            
            final_batch_example_inputs = [np.array(lst) for lst in batch_example_inputs_list]
            final_batch_outline_inputs_for_gen = np.array(batch_outline_inputs_for_gen)
            final_batch_target_outputs = np.array(batch_target_outputs)
            
            model_inputs = final_batch_example_inputs + [final_batch_outline_inputs_for_gen]
            yield model_inputs, final_batch_target_outputs

# --- Model Compilation & Training ---
def main():
    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)
    combined_model, _, _ = build_combined_model(
        img_shape=IMG_SHAPE, num_examples=NUM_EXAMPLES_CONDITION,
        example_latent_dim=EXAMPLE_LATENT_DIM, gen_filters_base=GEN_FILTERS_BASE
    )
    combined_model.summary()
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy']
    )
    train_gen = data_generator_fn(all_edge_images_np, all_outline_masks_np, BATCH_SIZE, NUM_EXAMPLES_CONDITION)
    steps_per_epoch = max(1, len(all_edge_images_np) // BATCH_SIZE)
    print(f"Starting training for {EPOCHS} epochs, with {steps_per_epoch} steps per epoch.")
    
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, MODEL_NAME), save_best_only=True, monitor='loss', verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=20, verbose=1, restore_best_weights=True
    )
    history = combined_model.fit(
        train_gen, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint_cb, early_stopping_cb]
    )
    print("Training finished.")
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss Over Epochs'); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend(); plt.savefig(os.path.join(MODEL_SAVE_DIR, "training_loss_plot.png"))

if __name__ == "__main__":
    main()
```

**4. `src/predict.py`**

*   **Purpose:** Loads a trained model and performs inference.
    *   Randomly samples 5 "edge detail" images and 1 "outline mask" image (or a few outline masks).
    *   Feeds these to the loaded model.
    *   Displays the input examples, the input outline, and the generated B&W image using Matplotlib.
*   **Note:** Similar to `train.py`, this needs updates to correctly sample from the two data directories for prediction. The current version assumes a simplified loading scheme.

```python
# src/predict.py
# !!! IMPORTANT: This version needs updates to properly load example edge_details
# !!! and test outline_masks from their respective directories.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob

# --- Configuration ---
IMG_DIM = 128
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
MODEL_PATH = "../trained_models/cat_pixelart_generator_128.keras" # Path to your saved .keras model

# Adjusted data directories for sampling prediction inputs
EDGE_DETAILS_DIR_FOR_SAMPLING = "../processed_data/edge_details/"
OUTLINE_MASKS_DIR_FOR_SAMPLING = "../processed_data/outline_masks/"


# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first.")
trained_model = tf.keras.models.load_model(MODEL_PATH)
print("Trained model loaded successfully.")

# --- Helper to load sample data for prediction (Modified) ---
def get_prediction_samples(edge_dir, outline_dir, num_examples_cond, num_outlines_to_test=1):
    edge_files = sorted(glob.glob(os.path.join(edge_dir, "*.npy")))
    outline_files = sorted(glob.glob(os.path.join(outline_dir, "*.npy")))

    if len(edge_files) < num_examples_cond:
        raise ValueError(f"Not enough edge detail images in {edge_dir} for sampling {num_examples_cond} examples.")
    if len(outline_files) < num_outlines_to_test:
        raise ValueError(f"Not enough outline mask images in {outline_dir} for sampling {num_outlines_to_test} outlines.")

    # Randomly pick distinct edge images for examples
    example_indices = random.sample(range(len(edge_files)), num_examples_cond)
    example_images = [np.load(edge_files[i]).reshape(IMG_SHAPE) for i in example_indices]
    
    # Randomly pick distinct outline images for testing
    outline_test_indices = random.sample(range(len(outline_files)), num_outlines_to_test)
    outline_images_for_test = [np.load(outline_files[i]).reshape(IMG_SHAPE) for i in outline_test_indices]
    
    return example_images, outline_images_for_test

# --- Prediction and Visualization ---
def predict_and_display(model, example_imgs_list, outline_img_list, threshold=0.5):
    num_outlines_to_show = len(outline_img_list)
    batch_example_inputs = [np.expand_dims(img, axis=0) for img in example_imgs_list]

    for i, outline_img_single in enumerate(outline_img_list):
        batch_outline_input = np.expand_dims(outline_img_single, axis=0)
        model_inputs_for_pred = batch_example_inputs + [batch_outline_input]
        generated_batch = model.predict(model_inputs_for_pred)
        generated_single = generated_batch[0]

        plt.figure(figsize=(15, 5))
        for j, ex_img in enumerate(example_imgs_list):
            plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, j + 1)
            plt.imshow(ex_img.squeeze(), cmap='gray_r'); plt.title(f"Ex {j+1}"); plt.axis('off')
        plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 1)
        plt.imshow(outline_img_single.squeeze(), cmap='gray_r'); plt.title(f"Input Outline {i+1}"); plt.axis('off')
        generated_display = (generated_single.squeeze() > threshold).astype(float)
        plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 2)
        plt.imshow(generated_display, cmap='gray_r'); plt.title(f"Generated {i+1}"); plt.axis('off')
        plt.suptitle(f"Prediction with Outline {i+1}"); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()

def main():
    num_outlines_to_test = 3
    try:
        example_images, outline_images_for_test = get_prediction_samples(
            EDGE_DETAILS_DIR_FOR_SAMPLING, OUTLINE_MASKS_DIR_FOR_SAMPLING,
            NUM_EXAMPLES_CONDITION, num_outlines_to_test
        )
    except ValueError as e:
        print(e); print("Try running prepare_data.py again or check dataset paths."); return

    print(f"Using {len(example_images)} example images and {len(outline_images_for_test)} outline(s) for prediction.")
    predict_and_display(trained_model, example_images, outline_images_for_test)

if __name__ == "__main__":
    main()
```

---

KEY DATA PREPARATION PARAMETERS (in `src/prepare_data.py`)

These parameters significantly affect the appearance of the "edge detail" images and should be tuned by inspecting the preview PNGs generated by `prepare_data.py`:

*   `GAUSSIAN_BLUR_KERNEL_SIZE`: e.g., `(3,3)`, `(5,5)`. Controls pre-Canny smoothing.
*   `GAUSSIAN_BLUR_SIGMAX`: Usually `0` (auto).
*   `CANNY_THRESHOLD1`: Lower Canny threshold.
*   `CANNY_THRESHOLD2`: Upper Canny threshold. These two are crucial for edge quality.
*   `DILATION_KERNEL_SIZE`: e.g., `(2,2)`, `(3,3)`. Controls thickening of detected edges.
*   `DILATION_ITERATIONS`: e.g., `1`, `2`. Number of times dilation is applied.

---