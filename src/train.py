# src/train.py
import tensorflow as tf
import numpy as np
import os
import random
import glob
import matplotlib.pyplot as plt
from model_def import build_combined_model  # Import from model_def.py
import cv2  # For potential resize failsafe

# --- Configuration ---
IMG_DIM = 128  # Must match TARGET_SIZE in prepare_data.py
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
BATCH_SIZE = 8  # Adjusted for 128x128, may need further tuning
EPOCHS = 200
LEARNING_RATE = 1e-4
EXAMPLE_LATENT_DIM = 64
GEN_FILTERS_BASE = 16

# Paths from your provided file
BASE_PROCESSED_DATA_DIR = "dataset/processed/grayscale_augmented_cat/" # Not directly used for sub-folders below, but kept for context
GRAYSCALE_CATS_DIR = "dataset/processed/grayscale_augmented_cat/grayscale_cats/" # Using this for grayscale images
OUTLINE_MASKS_DIR = "dataset/processed/grayscale_augmented_cat/outline_masks/"
MODEL_SAVE_DIR = "trained_models/"
MODEL_NAME = "cat_grayscale_augmented_generator.keras"

# --- Load Data ---
def load_paired_data(primary_data_dir, outline_masks_dir, img_dim_tuple, primary_suffix="_grayscale.npy"): # primary_suffix
    primary_files_pattern = os.path.join(primary_data_dir, f"*{primary_suffix}")
    all_primary_file_paths = sorted(glob.glob(primary_files_pattern))

    paired_primary_data = [] # e.g., grayscale cats
    paired_outline_data = []

    if not all_primary_file_paths:
        raise FileNotFoundError(
            f"No primary data files (e.g., *{primary_suffix}) found in {primary_data_dir}. Run prepare_data.py first.")

    for p_fpath in all_primary_file_paths:
        base_filename = os.path.basename(
            p_fpath).replace(primary_suffix, "")
        expected_outline_fname = f"{base_filename}_outline.npy"
        outline_fpath = os.path.join(outline_masks_dir, expected_outline_fname)

        if os.path.exists(outline_fpath):
            try:
                p_img = np.load(p_fpath)
                outline_img = np.load(outline_fpath)

                if p_img.shape != img_dim_tuple[:2]:
                    p_img = cv2.resize(
                        p_img, img_dim_tuple[:2], interpolation=cv2.INTER_AREA) # INTER_AREA better for grayscale
                if outline_img.shape != img_dim_tuple[:2]:
                    outline_img = cv2.resize(
                        outline_img, img_dim_tuple[:2], interpolation=cv2.INTER_NEAREST)

                paired_primary_data.append(p_img.reshape(img_dim_tuple))
                paired_outline_data.append(outline_img.reshape(img_dim_tuple))
            except Exception as e:
                print(
                    f"Warning: Error loading or processing pair for {base_filename}: {e}. Skipping.")
        else:
            print(
                f"Warning: Outline mask {expected_outline_fname} not found for primary data {os.path.basename(p_fpath)}. Skipping pair.")

    if not paired_primary_data:
        raise ValueError(
            "No valid pairs of primary data and outline masks found after loading.")

    return np.array(paired_primary_data), np.array(paired_outline_data)


all_grayscale_images_np, all_outline_images_np = load_paired_data(
    GRAYSCALE_CATS_DIR, OUTLINE_MASKS_DIR, IMG_SHAPE, primary_suffix="_grayscale.npy")
print(
    f"Loaded {len(all_grayscale_images_np)} paired grayscale cat and outline mask images.")

if len(all_grayscale_images_np) < NUM_EXAMPLES_CONDITION + 1:
    raise ValueError(
        f"Not enough images. Need at least {NUM_EXAMPLES_CONDITION + 1} pairs, found {len(all_grayscale_images_np)}")

# --- Data Generator ---
def data_generator_fn(primary_images_dataset, outline_images_dataset, batch_size, num_examples_cond): # Renamed primary_images_dataset
    num_total_images = len(primary_images_dataset)
    indices = np.arange(num_total_images)

    while True:
        np.random.shuffle(indices)

        for i in range(0, num_total_images, batch_size):
            batch_indices_overall = indices[i:i + batch_size]

            if len(batch_indices_overall) < batch_size:
                continue

            batch_example_inputs_list = [[] for _ in range(num_examples_cond)]
            batch_outline_inputs_for_generator = []
            batch_target_outputs = []

            for master_idx in batch_indices_overall:
                target_idx_in_dataset = master_idx

                possible_example_indices = list(range(num_total_images))
                possible_example_indices.remove(target_idx_in_dataset)

                if len(possible_example_indices) < num_examples_cond:
                    example_indices_for_instance = random.choices(possible_example_indices, k=num_examples_cond)
                else:
                    example_indices_for_instance = random.sample(possible_example_indices, num_examples_cond)

                # Load example images (primary data, e.g., grayscale cats)
                example_imgs = primary_images_dataset[example_indices_for_instance]

                # Load the outline mask that the generator should fill
                outline_input_for_gen = outline_images_dataset[target_idx_in_dataset]
                
                # The target output for the model is the PRIMARY DATA image corresponding to the input outline
                target_img_primary = primary_images_dataset[target_idx_in_dataset] 
                
                for k in range(num_examples_cond):
                    batch_example_inputs_list[k].append(example_imgs[k])

                batch_outline_inputs_for_generator.append(outline_input_for_gen)
                batch_target_outputs.append(target_img_primary)

            final_batch_example_inputs = [np.array(lst) for lst in batch_example_inputs_list]
            final_batch_outline_inputs_for_generator = np.array(batch_outline_inputs_for_generator)
            final_batch_target_outputs = np.array(batch_target_outputs)
            
            # Tensorflow's from_generator expects numpy arrays or Python native types.
            # It will handle the conversion to Tensors internally.
            model_inputs_np = tuple(final_batch_example_inputs + [final_batch_outline_inputs_for_generator])
            model_targets_np = final_batch_target_outputs
            
            yield model_inputs_np, model_targets_np

# --- Model Compilation & Training ---
def main():
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    combined_model, _, _ = build_combined_model(
        img_shape=IMG_SHAPE,
        num_examples=NUM_EXAMPLES_CONDITION,
        example_latent_dim=EXAMPLE_LATENT_DIM,
        gen_filters_base=GEN_FILTERS_BASE
    )
    combined_model.summary()
    
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(), # Or MeanSquaredError if grayscale values are not strictly 0/1
        metrics=['mae', 'accuracy'] # MAE is good for grayscale regression, accuracy for BCE-like scenario
    )
    
    # In train.py, after model compilation
    # Assuming outline_input_for_gen is (batch, H, W, 1) where 0 is cat, 1 is bg
    # and target_img_grayscale is the ground truth

    # You would need to pass the target outline mask to the loss function or calculate loss manually.
    # A custom training loop (model.train_on_batch) gives more flexibility here.
    # With model.fit(), you'd need a custom loss function:

    def masked_mse_loss(outline_mask_for_target): # This needs to be part of y_true or a separate input
        def loss(y_true_gs, y_pred_gs):
            mask = 1.0 - outline_mask_for_target # Mask is 1 where cat is, 0 where bg is
            squared_difference = tf.square(y_true_gs - y_pred_gs)
            masked_squared_difference = squared_difference * mask
            return tf.reduce_sum(masked_squared_difference) / tf.reduce_sum(mask) # Average over masked pixels
        return loss

    # This gets complicated with model.fit() data pipeline.
    # Simplest first step: Ensure your grayscale target images have a very consistent background (e.g., perfect 1.0)
    # and the model learns this from the data.

    train_gen_tf_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_fn( 
            all_grayscale_images_np, all_outline_images_np, BATCH_SIZE, NUM_EXAMPLES_CONDITION),
        output_signature=(
            tuple([tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32) for _ in range(NUM_EXAMPLES_CONDITION)] + \
                  [tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)]), # Use None for batch_size in signature
            tf.TensorSpec(shape=(None, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE) # Added prefetch for performance
    
    steps_per_epoch = len(all_grayscale_images_np) // BATCH_SIZE

    print(
        f"Starting training for {EPOCHS} epochs, with {steps_per_epoch} steps per epoch.")

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, MODEL_NAME),
        save_best_only=True, monitor='loss',
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    history = combined_model.fit(
        train_gen_tf_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_checkpoint_cb, early_stopping_cb]
    )

    combined_model.save(os.path.join(MODEL_SAVE_DIR, "final_" + MODEL_NAME))
    print("Training finished.")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "training_loss_plot_v2.png"))
    # plt.show()

if __name__ == "__main__":
    main()