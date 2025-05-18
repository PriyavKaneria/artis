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
EPOCHS = 500
LEARNING_RATE = 4e-4
EXAMPLE_LATENT_DIM = 64
GEN_FILTERS_BASE = 16

BASE_PROCESSED_DATA_DIR = "dataset/processed/grayscale_cat/" # Preserved path
GRAYSCALE_CATS_DIR = os.path.join(BASE_PROCESSED_DATA_DIR, "grayscale_cats/") # Updated
OUTLINE_MASKS_DIR = os.path.join(BASE_PROCESSED_DATA_DIR, "outline_masks/") # Preserved path
MODEL_SAVE_DIR = "trained_models/"                   # Preserved path
MODEL_NAME = "cat_grayscale_generator.keras" # Preserved path (consider _v3 if you want to keep old model)

# --- Load Data ---
def load_paired_data(grayscale_cats_dir, outline_masks_dir, img_dim_tuple):
    grayscale_files_pattern = os.path.join(grayscale_cats_dir, "*_grayscale.npy") # Updated pattern
    all_grayscale_file_paths = sorted(glob.glob(grayscale_files_pattern))

    paired_grayscale_data = []
    paired_outline_data = []

    if not all_grayscale_file_paths:
        raise FileNotFoundError(
            f"No grayscale cat files found in {grayscale_cats_dir}. Run prepare_data.py first.")

    for gs_fpath in all_grayscale_file_paths:
        base_filename = os.path.basename(
            gs_fpath).replace("_grayscale.npy", "") # Updated replace
        expected_outline_fname = f"{base_filename}_outline.npy"
        outline_fpath = os.path.join(outline_masks_dir, expected_outline_fname)

        if os.path.exists(outline_fpath):
            try:
                gs_img = np.load(gs_fpath)
                outline_img = np.load(outline_fpath)

                if gs_img.shape != img_dim_tuple[:2]:  # Check only H,W
                    gs_img = cv2.resize(
                        gs_img, img_dim_tuple[:2], interpolation=cv2.INTER_NEAREST) # INTER_AREA might be better for grayscale
                if outline_img.shape != img_dim_tuple[:2]:
                    outline_img = cv2.resize(
                        outline_img, img_dim_tuple[:2], interpolation=cv2.INTER_NEAREST)

                paired_grayscale_data.append(gs_img.reshape(img_dim_tuple))
                paired_outline_data.append(outline_img.reshape(img_dim_tuple))
            except Exception as e:
                print(
                    f"Warning: Error loading or processing pair for {base_filename}: {e}. Skipping.")
        else:
            print(
                f"Warning: Outline mask {expected_outline_fname} not found for grayscale cat {os.path.basename(gs_fpath)}. Skipping pair.")

    if not paired_grayscale_data:
        raise ValueError(
            "No valid pairs of grayscale cats and outline masks found after loading.")

    return np.array(paired_grayscale_data), np.array(paired_outline_data)


all_grayscale_images_np, all_outline_images_np = load_paired_data(
    GRAYSCALE_CATS_DIR, OUTLINE_MASKS_DIR, IMG_SHAPE) # Updated variable name
print(
    f"Loaded {len(all_grayscale_images_np)} paired B&W grayscale cat and outline mask images.")

if len(all_grayscale_images_np) < NUM_EXAMPLES_CONDITION + 1:
    raise ValueError(
        f"Not enough images. Need at least {NUM_EXAMPLES_CONDITION + 1} pairs, found {len(all_grayscale_images_np)}")

# --- Data Generator ---
def data_generator_fn(grayscale_images_dataset, outline_images_dataset, batch_size, num_examples_cond):
    num_total_images = len(grayscale_images_dataset)
    indices = np.arange(num_total_images)

    while True:
        np.random.shuffle(indices)

        for i in range(0, num_total_images, batch_size):
            batch_indices_overall = indices[i:i + batch_size]

            if len(batch_indices_overall) < batch_size: # Skip incomplete batches for from_generator
                continue

            batch_example_inputs_list = [[] for _ in range(num_examples_cond)]
            batch_outline_inputs_for_generator = []
            batch_target_outputs = [] # This will now store grayscale cat images

            for master_idx in batch_indices_overall:
                target_idx_in_dataset = master_idx

                possible_example_indices = list(range(num_total_images))
                possible_example_indices.remove(target_idx_in_dataset)

                if len(possible_example_indices) < num_examples_cond:
                    example_indices_for_instance = random.choices(possible_example_indices, k=num_examples_cond)
                else:
                    example_indices_for_instance = random.sample(possible_example_indices, num_examples_cond)

                # Load example images (grayscale cats)
                example_imgs = grayscale_images_dataset[example_indices_for_instance]

                # Load the outline mask that the generator should fill
                outline_input_for_gen = outline_images_dataset[target_idx_in_dataset]
                
                # The target output for the model is the GRAYSCALE CAT image corresponding to the input outline
                target_img_grayscale = grayscale_images_dataset[target_idx_in_dataset] 
                
                for k in range(num_examples_cond):
                    batch_example_inputs_list[k].append(example_imgs[k])

                batch_outline_inputs_for_generator.append(outline_input_for_gen)
                batch_target_outputs.append(target_img_grayscale) # Append the grayscale cat as target

            final_batch_example_inputs = [np.array(lst) for lst in batch_example_inputs_list]
            final_batch_outline_inputs_for_generator = np.array(batch_outline_inputs_for_generator)
            final_batch_target_outputs = np.array(batch_target_outputs)
            
            # Yield NumPy arrays. tf.data.Dataset.from_generator will handle tensor conversion.
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

    # Custom loss: BCE applied only within the target outline mask
    # The outline_mask is 0 for cat, 1 for background. We want loss where mask is 0 (cat).
    # So, we use (1.0 - outline_mask) which is 1 for cat, 0 for background.
    # This requires passing the outline_mask_of_target_image alongside y_true and y_pred.
    # Simpler: Just use standard BCE. The model should learn to generate white (1.0) for the background
    # if the target grayscale images consistently have a white background around the cat.
    # The input outline_mask to the generator helps it know where to *focus* its generation.
    
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(), # Or MeanSquaredError for grayscale
        metrics=['accuracy'] # 'mae' might be more interpretable for grayscale regression
    )

    train_gen_tf_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator_fn( # Use a lambda to pass arguments to the generator
            all_grayscale_images_np, all_outline_images_np, BATCH_SIZE, NUM_EXAMPLES_CONDITION),
        output_signature=(
            tuple([tf.TensorSpec(shape=(BATCH_SIZE, IMG_DIM, IMG_DIM, 1), dtype=tf.float32) for _ in range(NUM_EXAMPLES_CONDITION)] + \
                  [tf.TensorSpec(shape=(BATCH_SIZE, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)]),
            tf.TensorSpec(shape=(BATCH_SIZE, IMG_DIM, IMG_DIM, 1), dtype=tf.float32)
        )
    )
    
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
    plt.savefig(os.path.join(MODEL_SAVE_DIR, "training_loss_plot_v2.png")) # Preserved plot name
    # plt.show()

if __name__ == "__main__":
    main()