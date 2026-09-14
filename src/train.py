# src/train.py
import tensorflow as tf
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
import cv2 # For distanceTransform

# Import new model and custom layers
from model_def import (DistanceWeightedSoftSpatialLoss, build_full_hybrid_model, FiLMLayer, PatchEmbed, StyleEncoder, 
                       SpatialPlanner, AdaptiveBottleneck, AdaptiveGenerator)

# --- Configuration ---
IMG_DIM = 128
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
BATCH_SIZE = 2 # START VERY SMALL for this complex model on Mac. Increase if memory allows.
EPOCHS = 200   # Adjust as needed
LEARNING_RATE = 1e-4 # Start with this, might need tuning

# Model Hyperparameters (must match model_def.py for build_full_hybrid_model)
PATCH_SIZE_STYLE_DEF = 16 # Default from model_def.py StyleEncoder
EMBED_DIM_STYLE_DEF = 128
NUM_HEADS_STYLE_DEF = 4
NUM_TRANS_LAYERS_STYLE_DEF = 2
NUM_STYLE_TOKENS_DEF = 5
PLAN_CHANNELS_DEF = 64
NUM_ATTN_HEADS_PLAN_DEF = 4
GEN_FILTERS_BASE_DEF = 24 # Start with smaller generator

# Data paths
OUTPUT_DIR_BASE_PROCESSED = "dataset/processed/grayscale_augmented_cat_v3/"
GRAYSCALE_CATS_DIR = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "grayscale_cats/")
OUTLINE_MASKS_DIR = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "outline_masks/")
MODEL_SAVE_DIR = "trained_models/"
MODEL_NAME = "hybrid_generator_v1.keras" # New model name
PLOT_NAME = "training_hybrid_v1.png"

# --- Data Loading and Generator ---
# (load_paired_data_for_phase1 from your previous train.py should be adaptable)
# (It was already complex and good at finding pairs)
def load_dataset_items(grayscale_cats_dir, outline_masks_dir):
    gs_pattern = os.path.join(grayscale_cats_dir, "*_grayscale.npy")
    all_gs_paths = sorted(glob.glob(gs_pattern))
    
    outline_pattern = os.path.join(outline_masks_dir, "*_outline.npy")
    all_outline_paths = sorted(glob.glob(outline_pattern))

    # Create a dictionary mapping content_key_name to its actual grayscale path
    # content_key_name examples: "cat1", "cat1_aug1"
    content_key_to_actual_gs_path = {}
    for gs_path in all_gs_paths:
        gs_basename = os.path.basename(gs_path).replace("_grayscale.npy", "")
        # If gs_basename is "cat1_aug1_approx0", its true content is "cat1_aug1_grayscale.npy"
        # If gs_basename is "cat1_aug1", its true content is "cat1_aug1_grayscale.npy"
        # If gs_basename is "cat1", its true content is "cat1_grayscale.npy"
        content_key = gs_basename.split("_approx")[0] # "cat1" or "cat1_aug1"
        
        # We only store the path to the non-approximated grayscale content file
        if "_approx" not in gs_basename: 
            content_key_to_actual_gs_path[content_key] = gs_path
            
    print(f"Found {len(content_key_to_actual_gs_path)} unique content keys for grayscale images.")
    print(f"Found {len(all_gs_paths)} total grayscale images, {len(all_outline_paths)} outline masks.")

    dataset_items = []
    base_gs_paths_for_examples = [p for p in all_gs_paths if "_aug" not in p.split('/')[-1] and "_approx" not in p.split('/')[-1]]
    if not base_gs_paths_for_examples:
        print("Warning: No base (non-augmented, non-approx) grayscale images found for examples.")

    for input_outline_path in all_outline_paths:
        full_outline_basename = os.path.basename(input_outline_path).replace("_outline.npy", "")
        # This full_outline_basename is e.g. "cat1", "cat1_aug1", "cat1_approx0", "cat1_aug1_approx0"
        
        # Determine the content key associated with this outline (e.g., "cat1" or "cat1_aug1")
        content_key_for_this_outline = full_outline_basename.split("_approx")[0]

        if content_key_for_this_outline in content_key_to_actual_gs_path:
            target_grayscale_path = content_key_to_actual_gs_path[content_key_for_this_outline]
            # The input_outline_path IS the outline to be used for loss weighting as well.
            
            # Select example paths (from base non-augmented images)
            current_target_content_source_path = target_grayscale_path # Examples should ideally not be this exact file
            possible_example_paths = [p for p in base_gs_paths_for_examples if p != current_target_content_source_path]
            
            if len(possible_example_paths) < NUM_EXAMPLES_CONDITION:
                # If not enough unique, sample with replacement from all base images
                example_gs_paths = random.choices(base_gs_paths_for_examples, k=NUM_EXAMPLES_CONDITION) if base_gs_paths_for_examples else []
            else:
                example_gs_paths = random.sample(possible_example_paths, NUM_EXAMPLES_CONDITION)

            if len(example_gs_paths) == NUM_EXAMPLES_CONDITION:
                 dataset_items.append((
                    example_gs_paths,                # List of 5 example grayscale paths
                    input_outline_path,              # Path to the input outline for generator
                    target_grayscale_path,           # Path to the target grayscale image
                    input_outline_path               # Path to outline used for loss (same as input to gen)
                ))
        else:
            print(f"Warning: No matching grayscale content key found for outline {input_outline_path}. Content key derived: {content_key_for_this_outline}")

    if not dataset_items: raise ValueError("No dataset items created.")
    print(f"Created {len(dataset_items)} items for the data generator.")
    return dataset_items

dataset_items_for_generator = load_dataset_items(GRAYSCALE_CATS_DIR, OUTLINE_MASKS_DIR)

def data_generator_fn(dataset_items, batch_size, img_shape_tuple):
    num_total_items = len(dataset_items)
    indices = np.arange(num_total_items)

    while True:
        np.random.shuffle(indices)
        for i in range(0, num_total_items, batch_size):
            batch_item_indices = indices[i:i + batch_size]
            if len(batch_item_indices) < batch_size: continue

            batch_ex_gs_tensors_list = [[] for _ in range(NUM_EXAMPLES_CONDITION)]
            batch_input_ol_tensors = []
            batch_target_gs_tensors = []
            batch_loss_weight_ol_tensors = [] # For the outline used in loss

            for item_idx in batch_item_indices:
                ex_paths, input_ol_path, target_gs_path, loss_ol_path = dataset_items[item_idx]
                for k_ex in range(NUM_EXAMPLES_CONDITION):
                    batch_ex_gs_tensors_list[k_ex].append(np.load(ex_paths[k_ex]).reshape(img_shape_tuple))
                batch_input_ol_tensors.append(np.load(input_ol_path).reshape(img_shape_tuple))
                batch_target_gs_tensors.append(np.load(target_gs_path).reshape(img_shape_tuple))
                batch_loss_weight_ol_tensors.append(np.load(loss_ol_path).reshape(img_shape_tuple)) # This is 0 for obj, 1 for bg

            # Stack examples
            final_batch_examples_np = [np.array(lst_ex_np) for lst_ex_np in batch_ex_gs_tensors_list]
            final_batch_input_outlines_np = np.array(batch_input_ol_tensors)
            final_batch_target_grayscales_np = np.array(batch_target_gs_tensors)
            final_batch_loss_weight_outlines_np = np.array(batch_loss_weight_ol_tensors)
            
            model_inputs_tuple = tuple(final_batch_examples_np + [final_batch_input_outlines_np])
            
            # Package y_true for the custom loss: [target_grayscale, outline_for_loss_weights]
            # Both should be in [0,1] range. outline_for_loss_weights is 0 for object, 1 for background.
            y_true_for_loss = np.concatenate(
                [final_batch_target_grayscales_np, final_batch_loss_weight_outlines_np], axis=-1
            ) # Shape: (B, H, W, 2)
            
            yield model_inputs_tuple, y_true_for_loss


# --- Model and Training ---
def main():
    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR)

    model = build_full_hybrid_model(
        img_shape=IMG_SHAPE,
        num_examples=NUM_EXAMPLES_CONDITION,
        patch_size_style=PATCH_SIZE_STYLE_DEF,
        embed_dim_style=EMBED_DIM_STYLE_DEF,
        num_heads_style=NUM_HEADS_STYLE_DEF,
        num_transformer_layers_style=NUM_TRANS_LAYERS_STYLE_DEF,
        num_style_tokens=NUM_STYLE_TOKENS_DEF,
        plan_channels=PLAN_CHANNELS_DEF,
        num_attn_heads_plan=NUM_ATTN_HEADS_PLAN_DEF,
        gen_filters_base=GEN_FILTERS_BASE_DEF
    )
    model.summary(line_length=150)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=DistanceWeightedSoftSpatialLoss(), # Use the custom loss class instance
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae_on_pred')] # MAE on y_pred vs y_true (grayscale part)
    )

    # Output signature for generator
    # Inputs: ( (B,H,W,1) * 5_examples, (B,H,W,1)_outline )
    input_sig_examples = tuple([tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32) for _ in range(NUM_EXAMPLES_CONDITION)])
    input_sig_outline = tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)
    output_signature_inputs = tuple(list(input_sig_examples) + [input_sig_outline])
    
    # Targets: (B, H, W, 2) where ch0=target_gs, ch1=outline_for_loss
    output_signature_targets = tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 2), dtype=tf.float32)

    train_gen_tf_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_fn(dataset_items_for_generator, BATCH_SIZE, IMG_SHAPE),
        output_signature=(output_signature_inputs, output_signature_targets)
    ).prefetch(tf.data.AUTOTUNE)
    
    steps_per_epoch = len(dataset_items_for_generator) // BATCH_SIZE
    if steps_per_epoch == 0:
        raise ValueError(f"Steps_per_epoch is 0. Dataset items ({len(dataset_items_for_generator)}) vs BATCH_SIZE ({BATCH_SIZE}).")

    print(f"Starting training Hybrid model for {EPOCHS} epochs...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_SAVE_DIR, MODEL_NAME), save_best_only=True, monitor='loss', verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=35, verbose=1, restore_best_weights=True), # Increased patience
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=15, min_lr=1e-7, verbose=1) # Adjusted params
    ]
    history = model.fit(train_gen_tf_dataset, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    final_model_path_keras = os.path.join(MODEL_SAVE_DIR, "final_" + MODEL_NAME)
    model.save(final_model_path_keras) 
    print(f"Final Hybrid model saved to {final_model_path_keras}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Loss'); plt.title('Model Loss (Distance Weighted)'); plt.legend()
    if 'mae_on_pred' in history.history: # Note: MAE is on y_pred vs y_true (which has 2 channels)
        plt.subplot(1, 2, 2); plt.plot(history.history['mae_on_pred'], label='MAE'); plt.title('Model MAE'); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(MODEL_SAVE_DIR, PLOT_NAME))
    print(f"Training plots saved for Hybrid model.")

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    main()