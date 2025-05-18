# src/predict.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob

# --- Configuration ---
IMG_DIM = 128 # Must match training
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
MODEL_PATH = "trained_models/cat_grayscale_generator.keras" # Preserved path

BASE_PROCESSED_DATA_DIR_FOR_SAMPLING = "dataset/processed/grayscale_cat/" # Preserved path
GRAYSCALE_CATS_DIR_SAMPLING = os.path.join(BASE_PROCESSED_DATA_DIR_FOR_SAMPLING, "grayscale_cats/") # Updated
OUTLINE_MASKS_DIR_SAMPLING = os.path.join(BASE_PROCESSED_DATA_DIR_FOR_SAMPLING, "outline_masks/") # Preserved path

print(GRAYSCALE_CATS_DIR_SAMPLING, OUTLINE_MASKS_DIR_SAMPLING)

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first.")
trained_model = tf.keras.models.load_model(MODEL_PATH) # Allow loading custom objects if any
print("Trained model loaded successfully.")

# --- Helper to load sample data for prediction ---
def get_prediction_samples(grayscale_dir, outline_dir, num_examples, num_outlines_to_display, img_shape_tuple):
    grayscale_files_pattern = os.path.join(grayscale_dir, "*_grayscale.npy") # Updated
    all_grayscale_file_paths = sorted(glob.glob(grayscale_files_pattern))

    valid_base_filenames = []
    for gs_fpath in all_grayscale_file_paths:
        base = os.path.basename(gs_fpath).replace("_grayscale.npy", "") # Updated
        outline_fpath = os.path.join(outline_dir, f"{base}_outline.npy")
        if os.path.exists(outline_fpath):
            valid_base_filenames.append(base)
    
    if not valid_base_filenames:
        raise ValueError("No valid paired image sets found for sampling.")
    
    if len(valid_base_filenames) < num_examples + num_outlines_to_display:
        print(f"Warning: Not enough unique files ({len(valid_base_filenames)}) to satisfy num_examples ({num_examples}) + num_outlines ({num_outlines_to_display}) without overlap. Sampling may have overlaps.")


    # Sample example bases
    if len(valid_base_filenames) < num_examples:
        sampled_example_bases = random.choices(valid_base_filenames, k=num_examples) # Sample with replacement
    else:
        sampled_example_bases = random.sample(valid_base_filenames, num_examples)
    
    # Sample outline bases, try to make them different from examples if possible
    remaining_bases_for_outlines = list(set(valid_base_filenames) - set(sampled_example_bases))
    if len(remaining_bases_for_outlines) < num_outlines_to_display:
        # Not enough unique outlines different from examples, sample from all available
        if len(valid_base_filenames) < num_outlines_to_display:
             sampled_outline_bases = random.choices(valid_base_filenames, k=num_outlines_to_display)
        else:
            sampled_outline_bases = random.sample(valid_base_filenames, num_outlines_to_display)
    else:
        sampled_outline_bases = random.sample(remaining_bases_for_outlines, num_outlines_to_display)


    example_images_data = []
    for base in sampled_example_bases:
        gs_path = os.path.join(grayscale_dir, f"{base}_grayscale.npy") # Updated
        img = np.load(gs_path).reshape(img_shape_tuple)
        example_images_data.append(img)

    outline_images_for_testing = []
    for base in sampled_outline_bases:
        outline_path = os.path.join(outline_dir, f"{base}_outline.npy")
        img = np.load(outline_path).reshape(img_shape_tuple)
        outline_images_for_testing.append(img)
        
    return example_images_data, outline_images_for_testing

# --- Prediction and Visualization ---
def predict_and_display(model, example_grayscale_imgs_list, input_outline_mask_list, threshold=None): # Threshold might not be needed for grayscale
    num_outlines_to_show = len(input_outline_mask_list)
    
    batch_example_inputs = [np.expand_dims(img, axis=0) for img in example_grayscale_imgs_list]

    for i, outline_img_single in enumerate(input_outline_mask_list):
        batch_outline_input_for_generator = np.expand_dims(outline_img_single, axis=0)
        
        model_inputs_for_pred = batch_example_inputs + [batch_outline_input_for_generator]
        
        generated_batch = model.predict(model_inputs_for_pred)
        generated_single_grayscale = generated_batch[0] # This is now a grayscale image

        plt.figure(figsize=(18, 4)) 
        
        for j, ex_img in enumerate(example_grayscale_imgs_list):
            plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, j + 1)
            plt.imshow(ex_img.squeeze(), cmap='gray', vmin=0, vmax=1) # Display as grayscale
            plt.title(f"Ex Grayscale {j+1}") # Updated title
            plt.axis('off')
            
        plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 1)
        # Input outline is still B&W (0 for cat, 1 for bg), so gray_r is good here
        plt.imshow(outline_img_single.squeeze(), cmap='gray_r') 
        plt.title(f"Input Outline {i+1}")
        plt.axis('off')

        # Display generated grayscale image
        generated_display = generated_single_grayscale.squeeze()
        # if threshold is not None: # Apply threshold if you want to binarize for viewing
        #    generated_display = (generated_display > threshold).astype(float)
        
        plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 2)
        plt.imshow(generated_display, cmap='gray', vmin=0, vmax=1) # Display as grayscale
        plt.title(f"Generated Grayscale {i+1}") # Updated title
        plt.axis('off')
        
        plt.suptitle(f"Prediction - Input Outline {i+1}")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

def main():
    num_test_sets = 3
    try:
        example_images, outline_images_for_test = get_prediction_samples(
            GRAYSCALE_CATS_DIR_SAMPLING, 
            OUTLINE_MASKS_DIR_SAMPLING,
            NUM_EXAMPLES_CONDITION,
            num_test_sets,
            IMG_SHAPE
        )
    except Exception as e:
        print(f"Error getting prediction samples: {e}")
        return

    print(f"Using {len(example_images)} example grayscale images and {len(outline_images_for_test)} input outline masks for prediction.")
    predict_and_display(trained_model, example_images, outline_images_for_test)

if __name__ == "__main__":
    main()