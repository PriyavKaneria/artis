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

BASE_PROCESSED_DATA_DIR = "dataset/processed/pixel_art_cats/"
EDGE_DETAILS_DIR = "dataset/processed/pixel_art_cat/edge_details/"     # Relative to src/
OUTLINE_MASKS_DIR = "dataset/processed/pixel_art_cat/outline_masks/"   # Relative to src/
MODEL_SAVE_DIR = "trained_models/"                   # Relative to src/
MODEL_NAME = "cat_pixelart_generator.keras"

# --- Load Data ---


def load_paired_data(edge_details_dir, outline_masks_dir, img_dim_tuple):
    edge_files_pattern = os.path.join(edge_details_dir, "*_edgedetail.npy")
    all_edge_file_paths = sorted(glob.glob(edge_files_pattern))

    paired_edge_data = []
    paired_outline_data = []

    if not all_edge_file_paths:
        raise FileNotFoundError(
            f"No edge detail files found in {edge_details_dir}. Run prepare_data.py first.")

    for edge_fpath in all_edge_file_paths:
        base_filename = os.path.basename(
            edge_fpath).replace("_edgedetail.npy", "")
        expected_outline_fname = f"{base_filename}_outline.npy"
        outline_fpath = os.path.join(outline_masks_dir, expected_outline_fname)

        if os.path.exists(outline_fpath):
            try:
                edge_img = np.load(edge_fpath)
                outline_img = np.load(outline_fpath)

                if edge_img.shape != img_dim_tuple[:2]:  # Check only H,W
                    edge_img = cv2.resize(
                        edge_img, img_dim_tuple[:2], interpolation=cv2.INTER_NEAREST)
                if outline_img.shape != img_dim_tuple[:2]:
                    outline_img = cv2.resize(
                        outline_img, img_dim_tuple[:2], interpolation=cv2.INTER_NEAREST)

                paired_edge_data.append(edge_img.reshape(img_dim_tuple))
                paired_outline_data.append(outline_img.reshape(img_dim_tuple))
            except Exception as e:
                print(
                    f"Warning: Error loading or processing pair for {base_filename}: {e}. Skipping.")
        else:
            print(
                f"Warning: Outline mask {expected_outline_fname} not found for edge detail {os.path.basename(edge_fpath)}. Skipping pair.")

    if not paired_edge_data:
        raise ValueError(
            "No valid pairs of edge details and outline masks found after loading.")

    return np.array(paired_edge_data), np.array(paired_outline_data)


all_edge_images_np, all_outline_images_np = load_paired_data(
    EDGE_DETAILS_DIR, OUTLINE_MASKS_DIR, IMG_SHAPE)
print(
    f"Loaded {len(all_edge_images_np)} paired B&W edge detail and outline mask images.")

if len(all_edge_images_np) < NUM_EXAMPLES_CONDITION + 1:
    raise ValueError(
        f"Not enough images. Need at least {NUM_EXAMPLES_CONDITION + 1} pairs, found {len(all_edge_images_np)}")

# --- Data Generator ---
# Modify the data_generator_fn to ensure compatibility with TensorFlow's expectations


def data_generator_fn(edge_images_dataset, outline_images_dataset, batch_size, num_examples_cond):
    num_total_images = len(edge_images_dataset)
    indices = np.arange(num_total_images)

    while True:
        np.random.shuffle(indices)

        for i in range(0, num_total_images, batch_size):
            batch_indices_overall = indices[i:i + batch_size]

            # Skip the last batch if it's smaller than batch_size
            if len(batch_indices_overall) < batch_size:
                continue

            batch_example_inputs_list = [[] for _ in range(num_examples_cond)]
            batch_outline_inputs_for_generator = []
            batch_target_outputs = []

            for master_idx in batch_indices_overall:
                target_idx_in_dataset = master_idx

                # Sample 'num_examples_cond' distinct indices for examples, excluding the target index
                possible_example_indices = list(range(num_total_images))
                possible_example_indices.remove(target_idx_in_dataset)

                if len(possible_example_indices) < num_examples_cond:
                    example_indices_for_instance = random.choices(possible_example_indices, k=num_examples_cond)
                else:
                    example_indices_for_instance = random.sample(possible_example_indices, num_examples_cond)

                # Load example images (edge details)
                example_imgs = edge_images_dataset[example_indices_for_instance]

                # Load target image (edge detail of target outline image) and generator's outline input (also outline mask)
                target_img = edge_images_dataset[target_idx_in_dataset]
                target_img_outline = outline_images_dataset[target_idx_in_dataset]
                outline_input_for_gen = target_img_outline

                for k in range(num_examples_cond):
                    batch_example_inputs_list[k].append(example_imgs[k])

                batch_outline_inputs_for_generator.append(outline_input_for_gen)
                batch_target_outputs.append(target_img)

            final_batch_example_inputs = [np.array(lst) for lst in batch_example_inputs_list]
            final_batch_outline_inputs_for_generator = np.array(batch_outline_inputs_for_generator)
            final_batch_target_outputs = np.array(batch_target_outputs)

            # Convert to TensorFlow tensors
            model_inputs = [tf.convert_to_tensor(arr) for arr in final_batch_example_inputs] + [
                tf.convert_to_tensor(final_batch_outline_inputs_for_generator)]
            model_targets = tf.convert_to_tensor(final_batch_target_outputs)

            yield tuple(model_inputs), model_targets

# --- Model Compilation & Training ---


def main():
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # Build the combined model
    # The sub-models (encoder, generator) are not explicitly used in training here,
    # but can be saved separately if desired from the returned objects.
    combined_model, _, _ = build_combined_model(
        img_shape=IMG_SHAPE,
        num_examples=NUM_EXAMPLES_CONDITION,
        example_latent_dim=EXAMPLE_LATENT_DIM,
        gen_filters_base=GEN_FILTERS_BASE
    )
    combined_model.summary()

    # Compile the model
    # BinaryCrossentropy is suitable for 0/1 pixel values.
    # Could also use MeanSquaredError if outputs are consistently 0.0 and 1.0.
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    train_gen = tf.data.Dataset.from_generator(
        lambda: data_generator_fn(
            all_edge_images_np, all_outline_images_np, BATCH_SIZE, NUM_EXAMPLES_CONDITION),
        output_signature=(
            tuple([tf.TensorSpec(shape=(BATCH_SIZE, IMG_DIM, IMG_DIM, 1), dtype=tf.float32) for _ in range(NUM_EXAMPLES_CONDITION)] +
                  [tf.TensorSpec(shape=(BATCH_SIZE, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)]),
            tf.TensorSpec(shape=(BATCH_SIZE, IMG_DIM,
                                 IMG_DIM, 1), dtype=tf.float32)
        )
    )
    steps_per_epoch = len(all_edge_images_np) // BATCH_SIZE  # Use floor division

    print(
        f"Starting training for {EPOCHS} epochs, with {steps_per_epoch} steps per epoch.")

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_DIR, MODEL_NAME),
        # or 'val_loss' if you add a validation split
        save_best_only=True, monitor='loss',
        verbose=1
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='loss',  # or 'val_loss'
        patience=20,    # Num epochs with no improvement after which training will be stopped
        verbose=1,
        restore_best_weights=True
    )
    # Add TensorBoard if desired:
    # tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_SAVE_DIR, "logs"), histogram_freq=1)

    history = combined_model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        # Add tensorboard_cb here if used
        callbacks=[model_checkpoint_cb, early_stopping_cb]
        # validation_data=val_gen, # Add if you create a validation split and generator
        # validation_steps=val_steps_per_epoch
    )

    # Save the final model (even if early stopping restored best, this saves the potentially further trained one)
    # The ModelCheckpoint callback already saves the best one.
    combined_model.save(os.path.join(MODEL_SAVE_DIR, "final_" + MODEL_NAME))
    print("Training finished.")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    # if 'val_loss' in history.history:
    #    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "training_loss_plot_v2.png"))
    # plt.show()


if __name__ == "__main__":
    main()
