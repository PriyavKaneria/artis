# src/train.py
import tensorflow as tf
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
from model_def import build_combined_film_model, FiLMLayer # Import FiLMLayer for loading model
import cv2

# --- Configuration ---
IMG_DIM = 128  # Must match model_def.py and prepare_data.py
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
BATCH_SIZE = 4 # Reduced BATCH_SIZE due to larger model, adjust based on your Mac's RAM/VRAM
EPOCHS = 300   # Increased epochs for more complex model
LEARNING_RATE = 1e-4 # Might need to adjust this, 1e-4 is a common start
EXAMPLE_LATENT_DIM = 256 # Must match model_def.py
STYLE_VECTOR_DIM = 512   # Must match model_def.py
GEN_FILTERS_BASE = 24    # Start with a slightly smaller U-Net base: 24 or 32

# Paths from your previous file structure
OUTPUT_DIR_BASE_PROCESSED = "dataset/processed/grayscale_augmented_cat_v3/" # From latest prepare_data
GRAYSCALE_CATS_DIR = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "grayscale_cats/")
OUTLINE_MASKS_DIR = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "outline_masks/")
MODEL_SAVE_DIR = "trained_models/"
MODEL_NAME = "cat_generator_film_v3.keras" # New model name
PLOT_NAME = "training_loss_film_v3.png"

# --- Load Data ---
def load_paired_data(primary_data_dir, outline_masks_dir, img_dim_tuple, primary_suffix="_grayscale.npy"):
    primary_files_pattern = os.path.join(primary_data_dir, f"*{primary_suffix}")
    all_primary_file_paths = sorted(glob.glob(primary_files_pattern))

    paired_primary_data = []
    paired_outline_data = []

    if not all_primary_file_paths:
        raise FileNotFoundError(
            f"No primary data files (e.g., *{primary_suffix}) found in {primary_data_dir}. Run prepare_data.py first.")

    for p_fpath in all_primary_file_paths:
        base_filename_with_aug_approx = os.path.basename(p_fpath).replace(primary_suffix, "")
        # The outline filename should exactly match this base (including _augX_approxY if present)
        expected_outline_fname = f"{base_filename_with_aug_approx}_outline.npy"
        outline_fpath = os.path.join(outline_masks_dir, expected_outline_fname)

        if os.path.exists(outline_fpath):
            try:
                p_img = np.load(p_fpath)
                outline_img = np.load(outline_fpath)

                if p_img.shape != img_dim_tuple[:2]:
                    p_img = cv2.resize(p_img, img_dim_tuple[:2], interpolation=cv2.INTER_AREA)
                if outline_img.shape != img_dim_tuple[:2]:
                    outline_img = cv2.resize(outline_img, img_dim_tuple[:2], interpolation=cv2.INTER_NEAREST)

                paired_primary_data.append(p_img.reshape(img_dim_tuple))
                paired_outline_data.append(outline_img.reshape(img_dim_tuple))
            except Exception as e:
                print(f"Warning: Error loading or processing pair for {base_filename_with_aug_approx}: {e}. Skipping.")
        else:
            print(f"Warning: Outline mask {expected_outline_fname} not found for primary data {os.path.basename(p_fpath)}. Skipping pair.")

    if not paired_primary_data:
        raise ValueError("No valid pairs of primary data and outline masks found after loading.")

    return np.array(paired_primary_data), np.array(paired_outline_data)

all_grayscale_images_np, all_outline_images_np = load_paired_data(
    GRAYSCALE_CATS_DIR, OUTLINE_MASKS_DIR, IMG_SHAPE, primary_suffix="_grayscale.npy")
print(f"Loaded {len(all_grayscale_images_np)} paired grayscale cat and outline mask images.")

if len(all_grayscale_images_np) < NUM_EXAMPLES_CONDITION + 1: # Need enough for examples + 1 target
    raise ValueError(
        f"Not enough images. Need at least {NUM_EXAMPLES_CONDITION + 1} pairs, found {len(all_grayscale_images_np)}")

# --- Data Generator ---
def data_generator_fn(primary_images_dataset, outline_images_dataset, batch_size, num_examples_cond):
    num_total_images = len(primary_images_dataset)
    indices = np.arange(num_total_images)

    while True:
        np.random.shuffle(indices)
        for i in range(0, num_total_images, batch_size):
            batch_indices_overall = indices[i:i + batch_size]
            if len(batch_indices_overall) < batch_size: continue # Skip incomplete last batch

            batch_example_inputs_list = [[] for _ in range(num_examples_cond)]
            batch_outline_inputs_for_generator = []
            batch_target_outputs = []

            for master_idx_in_current_batch_indices in batch_indices_overall:
                # This master_idx is an index into the *shuffled* 'indices' array,
                # which then points to the original dataset position.
                # For simplicity, let's directly use master_idx_in_current_batch_indices as the target_idx.
                target_idx_in_dataset = master_idx_in_current_batch_indices

                possible_example_indices = list(range(num_total_images))
                if target_idx_in_dataset in possible_example_indices: # Ensure target is not picked as example
                    possible_example_indices.remove(target_idx_in_dataset)
                
                if len(possible_example_indices) < num_examples_cond: # Should not happen if dataset is large enough
                    example_indices_for_instance = random.choices(possible_example_indices, k=num_examples_cond)
                else:
                    example_indices_for_instance = random.sample(possible_example_indices, num_examples_cond)

                example_imgs = primary_images_dataset[example_indices_for_instance]
                outline_input_for_gen = outline_images_dataset[target_idx_in_dataset]
                target_img_primary = primary_images_dataset[target_idx_in_dataset] 
                
                for k_ex in range(num_examples_cond):
                    batch_example_inputs_list[k_ex].append(example_imgs[k_ex])
                batch_outline_inputs_for_generator.append(outline_input_for_gen)
                batch_target_outputs.append(target_img_primary)

            final_batch_example_inputs_np = [np.array(lst) for lst in batch_example_inputs_list]
            final_batch_outline_inputs_np = np.array(batch_outline_inputs_for_generator)
            final_batch_target_outputs_np = np.array(batch_target_outputs)
            
            model_inputs_np_tuple = tuple(final_batch_example_inputs_np + [final_batch_outline_inputs_np])
            
            yield model_inputs_np_tuple, final_batch_target_outputs_np

# --- Model Compilation & Training ---
def main():
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # Build the combined FiLM model
    combined_model = build_combined_film_model(
        img_shape=IMG_SHAPE,
        num_examples=NUM_EXAMPLES_CONDITION,
        example_latent_dim=EXAMPLE_LATENT_DIM,
        style_vector_dim=STYLE_VECTOR_DIM,
        gen_filters_base=GEN_FILTERS_BASE # Use configured base filters
    )
    combined_model.summary(line_length=120)

    # Using Mean Squared Error for grayscale image regression (0-1 values)
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(), 
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')]
    )

    # Create tf.data.Dataset from generator
    # output_signature must match the structure and dtypes yielded by the generator
    output_signature_inputs_tuple = tuple(
        [tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32) for _ in range(NUM_EXAMPLES_CONDITION)] + \
        [tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)]
    )
    output_signature_targets = tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)

    train_gen_tf_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_fn(
            all_grayscale_images_np, all_outline_images_np, BATCH_SIZE, NUM_EXAMPLES_CONDITION),
        output_signature=(output_signature_inputs_tuple, output_signature_targets)
    ).prefetch(tf.data.AUTOTUNE)
    
    steps_per_epoch = len(all_grayscale_images_np) // BATCH_SIZE
    if steps_per_epoch == 0:
        raise ValueError(f"steps_per_epoch is 0. Dataset size ({len(all_grayscale_images_np)}) might be smaller than BATCH_SIZE ({BATCH_SIZE}).")

    print(f"Starting training for {EPOCHS} epochs, with {steps_per_epoch} steps per epoch (Batch Size: {BATCH_SIZE}).")

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, MODEL_NAME),
        save_best_only=True, monitor='loss', verbose=1 # Monitor training loss
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=30, verbose=1, restore_best_weights=True # Increased patience
    )
    # ReduceLROnPlateau can be helpful if loss stagnates
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1
    )

    history = combined_model.fit(
        train_gen_tf_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    # Save the final model explicitly (after EarlyStopping might have restored best)
    final_model_path = os.path.join(MODEL_SAVE_DIR, "final_" + MODEL_NAME.replace(".keras", ".h5")) # Save as H5 as well
    combined_model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    # Also save in Keras native format if preferred
    # combined_model.save(os.path.join(MODEL_SAVE_DIR, "final_" + MODEL_NAME), save_format="keras")


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    if 'mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, PLOT_NAME))
    print(f"Training plots saved to {os.path.join(MODEL_SAVE_DIR, PLOT_NAME)}")


if __name__ == "__main__":
    # Ensure TensorFlow uses GPU if available (especially on Mac with Metal)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # tf.debugging.set_log_device_placement(True) # For debugging device placement
    main()