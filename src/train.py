# src/train.py
import tensorflow as tf
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
import cv2 # For distanceTransform

# Import new model and custom layers if any
from model_def import build_combined_model_phase1, PatchEmbed, StyleEncoder # FiLMLayer not used yet by placeholder

# --- Configuration ---
IMG_DIM = 128
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
BATCH_SIZE = 4 # Start small due to potentially larger model / more complex data path
EPOCHS = 100   # Start with fewer epochs for initial testing of new setup
LEARNING_RATE = 1e-4

# StyleEncoder HParams (must match model_def.py for build_combined_model_phase1)
PATCH_SIZE_STYLE_ENC = 16
EMBED_DIM_STYLE_ENC = 128
NUM_HEADS_STYLE_ENC = 4
NUM_TRANS_LAYERS_STYLE_ENC = 2
NUM_STYLE_TOKENS_ENC = 5
# Placeholder U-Net HParams
GEN_FILTERS_BASE_PLACEHOLDER = 32

# Data paths (using your existing structure)
OUTPUT_DIR_BASE_PROCESSED = "dataset/processed/grayscale_augmented_cat_v3/" # From latest prepare_data
GRAYSCALE_CATS_DIR = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "grayscale_cats/")
OUTLINE_MASKS_DIR = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "outline_masks/")
MODEL_SAVE_DIR = "trained_models/"
MODEL_NAME = "cat_generator_style_encoder_v1.keras" # New model name for this phase
PLOT_NAME = "training_loss_style_encoder_v1.png"

# --- Distance Weighted Soft Spatial Loss ---
# Sigma controls flexibility: higher sigma = softer boundary constraint
# Pixels > 3*sigma away from boundary get almost full weight
# Pixels at boundary get exp(-0/sigma)=1 if dist=0 from boundary, but dist is to *nearest bg*, so this needs care
# Better: dist is to *nearest fg* for bg pixels, and *nearest bg* for fg pixels.
# For this loss, we care about the target outline mask.
# target_outline_mask_for_loss: 0 for object, 1 for background

DISTANCE_LOSS_SIGMA = 3.0 # Pixels, controls the falloff

def distance_weighted_soft_spatial_loss(y_true_grayscale, y_pred_grayscale, target_outline_mask_for_loss_0_obj_1_bg):
    # target_outline_mask_for_loss: (B, H, W, 1), where 0 is object, 1 is background.
    # We want high weights *inside* the object region defined by this mask.
    
    # 1. Create a binary mask where object region is 1, background is 0
    object_region_mask = 1.0 - target_outline_mask_for_loss_0_obj_1_bg # (B, H, W, 1), object=1, bg=0

    # 2. Compute distance transform: distance to the nearest zero (background) pixel from each foreground pixel
    # OpenCV's distanceTransform expects single channel uint8 image.
    # We need to do this per batch item. tf.py_function or tf.map_fn can be used.
    
    def compute_distance_weights_for_batch(batch_object_region_mask_uint8):
        batch_weights = []
        for i in range(tf.shape(batch_object_region_mask_uint8)[0]): # Loop over batch
            single_mask_uint8 = batch_object_region_mask_uint8[i, ..., 0].numpy() # Get (H,W) numpy array
            
            # Distance from each non-zero pixel to the closest zero pixel
            dist_transform = cv2.distanceTransform(single_mask_uint8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            
            # Weighting: exp(-distance_to_boundary / sigma) - but this makes boundary high, center low.
            # We want center high, boundary lower but still present.
            # Alternative: Use the object_region_mask directly for hard masking,
            # or a more complex weighting.
            # For now, let's use: weight = 1 inside object, 0.1 near boundary, 0 outside.
            # This is simpler than precise exp(-dist/sigma) for a first pass.
            
            # More robust: max_dist = np.max(dist_transform)
            # weights = np.exp(-(max_dist - dist_transform) / (sigma_dist_falloff_inside * max_dist + 1e-6))
            # This makes center high, edge of object lower.
            
            # Let's try the paper's idea: exp(-distance_from_true_boundary / sigma)
            # The "distance" here should be distance from the *true boundary of the object defined by the mask*.
            # Pixels deep inside the mask should have high weight.
            # Pixels near the boundary (small distance) have high weight.
            # Pixels outside the mask should have zero or very low weight.

            # Weighting based on distance transform from boundary.
            # dist_transform value is 0 at boundary, increases inwards.
            # We want high weight for small distances (near boundary AND inside)
            # and also high for large distances (deep inside).
            # This is tricky. Let's use the paper's intention: "weight loss by exp(-distance/sigma) near boundaries"
            # This implies distance is from the *input approximate outline's boundary*.
            # And we want pixels *inside* this outline to have higher weight.

            # For a pixel `p`, let `d(p)` be its distance to the *boundary* of the *input outline*.
            # Loss weight `w(p) = exp(-d(p)/sigma)` if `p` is near boundary.
            # Full weight if `p` is deep inside. Low/zero weight if `p` is outside.

            # Simpler interpretation:
            # 1. Mask where loss is applied: inside the `target_outline_mask_for_loss`.
            # 2. Within this mask, weight pixels near boundary less.
            
            # Let's use: weight = 1 for pixels inside the main object area.
            # We will achieve the "softness" by training with *approximate* outlines as input,
            # while the target grayscale and this loss_mask are based on the *precise* outline
            # that corresponds to the target grayscale.
            
            # The `object_region_mask` (0 where bg, 1 where object) is already a good start
            # for "full reconstruction loss in the outline's interior".
            # The "flexibility" comes from the fact that the input outline to the U-Net
            # can be approximate.
            
            # For now, let's use a simple binary mask derived from target_outline_mask_for_loss
            # This mask is 1.0 where the object is, 0.0 where background is.
            weights = single_mask_uint8.astype(np.float32) / 255.0 # Object=1, BG=0
            batch_weights.append(weights)
        return tf.expand_dims(tf.stack(batch_weights), axis=-1) # (B, H, W, 1)

    # Convert object_region_mask (float32, 0.0 or 1.0) to uint8 (0 or 255) for OpenCV
    object_region_mask_uint8 = tf.cast(object_region_mask * 255.0, tf.uint8)
    
    # Use tf.py_function to wrap the numpy/OpenCV code
    # This is the `loss_application_mask`
    loss_weights = tf.py_function(
        func=compute_distance_weights_for_batch,
        inp=[object_region_mask_uint8],
        Tout=tf.float32
    )
    loss_weights.set_shape(target_outline_mask_for_loss_0_obj_1_bg.shape) # Critical for graph mode

    # Standard MSE, but weighted
    pixel_wise_mse = tf.square(y_true_grayscale - y_pred_grayscale)
    weighted_pixel_wise_mse = pixel_wise_mse * loss_weights
    
    # Average loss over the pixels where weight > 0
    sum_weighted_mse = tf.reduce_sum(weighted_pixel_wise_mse, axis=[1,2,3])
    sum_weights = tf.reduce_sum(loss_weights, axis=[1,2,3]) + 1e-6 # Avoid division by zero
    
    mean_loss_per_sample = sum_weighted_mse / sum_weights
    return tf.reduce_mean(mean_loss_per_sample) # Average over batch


# --- Data Loading and Generator ---
def load_paired_data_for_phase1(grayscale_cats_dir, outline_masks_dir, img_dim_tuple):
    # Grayscale paths (these are unique content files, some base, some augmented)
    # e.g., cat1_grayscale.npy, cat1_aug1_grayscale.npy
    grayscale_files_pattern = os.path.join(grayscale_cats_dir, "*_grayscale.npy")
    all_grayscale_file_paths = sorted(glob.glob(grayscale_files_pattern))

    # Outline paths (these include base, _augX, _approxY variations)
    # e.g., cat1_outline.npy, cat1_aug1_outline.npy, cat1_aug1_approx0_outline.npy
    outline_files_pattern = os.path.join(outline_masks_dir, "*_outline.npy")
    all_outline_file_paths = sorted(glob.glob(outline_files_pattern))

    dataset_items = [] # List of tuples: (example_grayscale_paths_list, input_outline_path, target_grayscale_path, target_precise_outline_path_for_loss)

    # Create a mapping from a base_content_name (e.g. "cat1" or "cat1_aug1")
    # to its grayscale path and precise outline path.
    content_to_paths = {}
    for gs_path in all_grayscale_file_paths:
        # If gs_path is "dataset/.../cat1_aug1_approx0_grayscale.npy", its content_base is "cat1_aug1"
        # Its "true" content is "cat1_aug1_grayscale.npy".
        # Its "true" precise outline is "cat1_aug1_outline.npy".
        
        # The files in GRAYSCALE_CATS_DIR should be:
        # cat1_grayscale.npy (base content for cat1)
        # cat1_aug1_grayscale.npy (augmented content for cat1)
        # cat1_approx0_grayscale.npy (copy of cat1_grayscale.npy)
        # cat1_aug1_approx0_grayscale.npy (copy of cat1_aug1_grayscale.npy)
        
        # Let's simplify: find the "root" content for each grayscale file.
        # e.g., for "cat1_aug1_approx0_grayscale.npy", root is "cat1_aug1"
        # for "cat1_grayscale.npy", root is "cat1"
        
        gs_basename = os.path.basename(gs_path).replace("_grayscale.npy", "")
        parts = gs_basename.split("_approx")
        content_key_name = parts[0] # This is "cat1" or "cat1_aug1" etc.

        if content_key_name not in content_to_paths:
            # Find the actual grayscale file for this content key (without _approx)
            actual_content_gs_path = os.path.join(grayscale_cats_dir, f"{content_key_name}_grayscale.npy")
            # Find the precise outline for this content key
            precise_outline_for_content_path = os.path.join(outline_masks_dir, f"{content_key_name}_outline.npy")

            if os.path.exists(actual_content_gs_path) and os.path.exists(precise_outline_for_content_path):
                content_to_paths[content_key_name] = {
                    "grayscale_path": actual_content_gs_path,
                    "precise_outline_path": precise_outline_for_content_path
                }
                
    print(f"Found {len(content_to_paths)} unique content keys in grayscale files.")
    # print(all_grayscale_file_paths)
    
    # Now, iterate through all available outline files (base, aug, approx)
    for input_outline_path in all_outline_file_paths:
        outline_basename_full = os.path.basename(input_outline_path).replace("_outline.npy", "") # e.g. "cat1_aug1_approx0"
        
        # Determine the content key (e.g. "cat1_aug1") associated with this outline
        content_key_for_this_outline = outline_basename_full.split("_approx")[0]

        if content_key_for_this_outline in content_to_paths:
            target_grayscale_path = content_to_paths[content_key_for_this_outline]["grayscale_path"]
            target_precise_outline_path_for_loss = content_to_paths[content_key_for_this_outline]["precise_outline_path"]
            
            # For examples, pick randomly from *base* (non-augmented) grayscale images
            # These are grayscale files that do NOT contain "_aug" or "_approx"
            base_grayscale_example_pool = [
                p for p in all_grayscale_file_paths if "_aug" not in p.split('/')[-1] and "_approx" not in p.split('/')[-1]
            ]
            # print(f"Base example pool size: {len(base_grayscale_example_pool)}")
            if not base_grayscale_example_pool: continue # Should not happen if data is prepared

            # Ensure examples are different from the target's content source
            current_target_content_source_path = target_grayscale_path
            
            possible_example_paths = [p for p in base_grayscale_example_pool if p != current_target_content_source_path]
            
            num_examples_to_pick = min(NUM_EXAMPLES_CONDITION, len(possible_example_paths))
            if num_examples_to_pick < NUM_EXAMPLES_CONDITION and base_grayscale_example_pool: # If not enough unique, pick with replacement from all base
                 example_grayscale_paths_list = random.choices(base_grayscale_example_pool, k=NUM_EXAMPLES_CONDITION)
            elif possible_example_paths:
                 example_grayscale_paths_list = random.sample(possible_example_paths, num_examples_to_pick)
                 # Fill up if needed by resampling from all base
                 while len(example_grayscale_paths_list) < NUM_EXAMPLES_CONDITION and base_grayscale_example_pool:
                     example_grayscale_paths_list.append(random.choice(base_grayscale_example_pool))
            else: # Very small dataset
                example_grayscale_paths_list = random.choices(base_grayscale_example_pool, k=NUM_EXAMPLES_CONDITION)


            if len(example_grayscale_paths_list) == NUM_EXAMPLES_CONDITION:
                 dataset_items.append((
                    example_grayscale_paths_list, 
                    input_outline_path, 
                    target_grayscale_path,
                    target_precise_outline_path_for_loss # This outline is used for the loss mask
                ))
        else:
            print(f"Warning: Could not find matching content for outline {input_outline_path}. Skipping.")

    if not dataset_items:
        raise ValueError("No valid dataset items created. Check data preparation and paths.")
    
    print(f"Created {len(dataset_items)} items for the data generator.")
    return dataset_items


dataset_items_for_generator = load_paired_data_for_phase1(GRAYSCALE_CATS_DIR, OUTLINE_MASKS_DIR, IMG_SHAPE)

def data_generator_fn(dataset_items, batch_size, img_shape_tuple):
    num_total_items = len(dataset_items)
    indices = np.arange(num_total_items)

    while True:
        np.random.shuffle(indices)
        for i in range(0, num_total_items, batch_size):
            batch_item_indices = indices[i:i + batch_size]
            if len(batch_item_indices) < batch_size: continue

            batch_example_tensors_list = [[] for _ in range(NUM_EXAMPLES_CONDITION)]
            batch_input_outline_tensors = []
            batch_target_grayscale_tensors = []
            batch_target_precise_outline_for_loss_tensors = []

            for item_idx in batch_item_indices:
                ex_paths, input_ol_path, target_gs_path, target_precise_ol_path = dataset_items[item_idx]
                
                # Load examples
                for k_ex in range(NUM_EXAMPLES_CONDITION):
                    ex_img = np.load(ex_paths[k_ex]).reshape(img_shape_tuple)
                    batch_example_tensors_list[k_ex].append(ex_img)
                
                # Load input outline for generator
                input_ol_img = np.load(input_ol_path).reshape(img_shape_tuple)
                batch_input_outline_tensors.append(input_ol_img)

                # Load target grayscale image
                target_gs_img = np.load(target_gs_path).reshape(img_shape_tuple)
                batch_target_grayscale_tensors.append(target_gs_img)

                # Load precise outline for loss weighting
                target_precise_ol_img = np.load(target_precise_ol_path).reshape(img_shape_tuple)
                batch_target_precise_outline_for_loss_tensors.append(target_precise_ol_img)

            # Stack examples into batch tensors
            final_batch_examples = [np.array(lst_ex) for lst_ex in batch_example_tensors_list]
            final_batch_input_outlines = np.array(batch_input_outline_tensors)
            final_batch_target_grayscales = np.array(batch_target_grayscale_tensors)
            final_batch_target_loss_outlines = np.array(batch_target_precise_outline_for_loss_tensors)
            
            # Model inputs: list of example image batches + input_outline batch
            model_inputs_tuple = tuple(final_batch_examples + [final_batch_input_outlines])
            
            # For the custom loss, y_true needs to package both target_gs and target_loss_outline
            # Keras model.compile(loss=...) expects loss(y_true, y_pred).
            # So, y_true needs to contain everything needed by the loss function besides y_pred.
            # We'll pass a tuple/dict for y_true if the loss function can handle it,
            # or handle it via a custom training loop.
            # For now, the loss function is defined to take target_outline_mask_for_loss as an extra arg.
            # This structure won't work directly with model.fit() unless the loss is a class that holds this.
            
            # Let's redefine the generator to yield (X, y_true_grayscale, loss_weight_mask_outline)
            # And then the loss function will be loss(y_true_grayscale, y_pred_grayscale, loss_weight_mask=loss_weight_mask_outline)
            # This requires a model.add_loss or custom training loop.

            # Simpler for now: the loss function will get `final_batch_target_grayscales` as y_true.
            # We need to pass `final_batch_target_loss_outlines` to it somehow.
            # This will be handled by making the loss function a class or using add_loss.
            # For direct use with model.compile, the loss must be loss(y_true, y_pred).
            # Let's assume for this phase, our distance_weighted_soft_spatial_loss will use a
            # globally accessible or fixed sigma, and it will compute the weight mask *from y_true if y_true is the outline mask*.
            # This isn't ideal.
            
            # **REVISION for standard model.fit()**:
            # The y_true yielded by the generator will be the target_grayscale_image.
            # The input_outline_for_generator will be used *inside* the loss function
            # to calculate the distance transform weights. This makes the loss function
            # depend on one of the inputs `X`, which is not standard for Keras `loss(y_true, y_pred)`.

            # **Correct approach for model.fit() with custom loss needing extra data:**
            # The generator yields (X_dict, y_true_grayscale).
            # X_dict includes 'outline_input' and 'outline_for_loss_calc'.
            # The loss function is a class that gets compiled with the model.
            # In its __call__(self, y_true, y_pred), it can access `y_pred.graph.get_tensor_by_name('outline_for_loss_calc_input_tensor_name:0')`
            # This is getting complex.

            # **Easiest path for now: The `target_outline_mask_for_loss` is passed AS PART OF X**
            # And the loss function retrieves it from X. This implies changing model structure to accept it.
            # OR, simpler: the loss function uses the *generator's input outline* to calculate weights.
            # This makes sense: weight loss based on the outline given to the generator.
            
            # Yield: ( (example_batches..., input_outline_batch), target_grayscale_batch )
            # The loss function will take `input_outline_batch` from `X` (model inputs) implicitly.
            # This requires the loss function to be able to access model inputs.
            # This is NOT how standard Keras losses work.

            # **FINAL DECISION FOR THIS ITERATION (simplest for model.fit):**
            # The `distance_weighted_soft_spatial_loss` will take an *additional argument*
            # `input_outline_for_loss_weighting` which we will supply from the *input to the generator U-Net*.
            # This means our loss function needs to be wrapped or become a model metric that has access.
            # For `model.compile(loss=...)`, the loss signature must be `loss(y_true, y_pred)`.
            #
            # We will use `add_loss` layer in the model itself.
            
            # For this training script to run with the `distance_weighted_soft_spatial_loss` as defined
            # (taking 3 args), we need a custom training loop.
            # To use model.fit(), we need to re-think.

            # Let's go with a simplified loss for this step to get StyleEncoder running.
            # We will use standard MSE, and the "softness" will come from the varied approximate outlines.
            # The refined distance-weighted loss will be for the *next* iteration if this isn't enough.
            # This means `target_precise_outline_path_for_loss` is not strictly used by the loss *yet*.

            yield model_inputs_tuple, final_batch_target_grayscales


# --- Model and Training ---
def main():
    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)

    model = build_combined_model_phase1(
        img_shape=IMG_SHAPE,
        num_examples=NUM_EXAMPLES_CONDITION,
        patch_size_style=PATCH_SIZE_STYLE_ENC,
        embed_dim_style=EMBED_DIM_STYLE_ENC,
        num_heads_style=NUM_HEADS_STYLE_ENC,
        num_transformer_layers_style=NUM_TRANS_LAYERS_STYLE_ENC,
        num_style_tokens=NUM_STYLE_TOKENS_ENC,
        gen_filters_base=GEN_FILTERS_BASE_PLACEHOLDER
    )
    model.summary(line_length=120)

    # For this iteration, using simple MSE. The distance-weighted loss needs add_loss or custom loop.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(), # Placeholder simple loss for now
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )

    output_signature_inputs_tuple = tuple(
        [tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32) for _ in range(NUM_EXAMPLES_CONDITION)] + \
        [tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)]
    )
    output_signature_targets = tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)

    train_gen_tf_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_fn(dataset_items_for_generator, BATCH_SIZE, IMG_SHAPE),
        output_signature=(output_signature_inputs_tuple, output_signature_targets)
    ).prefetch(tf.data.AUTOTUNE)
    
    steps_per_epoch = len(dataset_items_for_generator) // BATCH_SIZE
    if steps_per_epoch == 0:
        raise ValueError(f"Steps_per_epoch is 0. Dataset items ({len(dataset_items_for_generator)}) vs BATCH_SIZE ({BATCH_SIZE}).")

    print(f"Starting training Phase 1 model for {EPOCHS} epochs...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_DIR, MODEL_NAME), save_best_only=True, monitor='loss', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=25, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(train_gen_tf_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_" + MODEL_NAME)
    model.save(final_model_path) # Keras native format
    print(f"Final Phase 1 model saved to {final_model_path}")

    # Plotting (same as before)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Loss'); plt.title('Model Loss (MSE)'); plt.legend()
    if 'mae' in history.history:
        plt.subplot(1, 2, 2); plt.plot(history.history['mae'], label='MAE'); plt.title('Model MAE'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(MODEL_SAVE_DIR, PLOT_NAME))
    print(f"Training plots saved for Phase 1 model.")

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()