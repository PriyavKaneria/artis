# src/predict.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob
# from prepare_data import create_filled_outline_mask, create_bnw_edge_detail_image, TARGET_SIZE # If processing new inputs on the fly

# --- Configuration ---
IMG_DIM = 128 # Must match training
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5
MODEL_PATH = "trained_models/final_cat_pixelart_generator.keras" # Path to your new saved model

BASE_PROCESSED_DATA_DIR_FOR_SAMPLING = "dataset/processed/pixel_art_cat/"
EDGE_DETAILS_DIR_SAMPLING = os.path.join(BASE_PROCESSED_DATA_DIR_FOR_SAMPLING, "edge_details/")
OUTLINE_MASKS_DIR_SAMPLING = os.path.join(BASE_PROCESSED_DATA_DIR_FOR_SAMPLING, "outline_masks/")

print(EDGE_DETAILS_DIR_SAMPLING, OUTLINE_MASKS_DIR_SAMPLING)

# --- Load Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first.")
trained_model = tf.keras.models.load_model(MODEL_PATH)
print("Trained model loaded successfully.")

# --- Helper to load sample data for prediction ---
def get_prediction_samples(edge_dir, outline_dir, num_examples, num_outlines_to_display, img_shape_tuple):
    edge_files_pattern = os.path.join(edge_dir, "*_edgedetail.npy")
    all_edge_file_paths = sorted(glob.glob(edge_files_pattern))

    valid_base_filenames = []
    for edge_fpath in all_edge_file_paths:
        base = os.path.basename(edge_fpath).replace("_edgedetail.npy", "")
        outline_fpath = os.path.join(outline_dir, f"{base}_outline.npy")
        if os.path.exists(outline_fpath):
            valid_base_filenames.append(base)
    
    if not valid_base_filenames:
        raise ValueError("No valid paired image sets found for sampling.")
    
    # Ensure enough unique samples if num_examples + num_outlines_to_display > len(valid_base_filenames)
    # For simplicity, we sample with replacement if needed, or cap at available unique files
    
    num_unique_needed_for_examples = min(num_examples, len(valid_base_filenames))
    sampled_example_bases = random.sample(valid_base_filenames, num_unique_needed_for_examples)
    # If fewer than num_examples unique files, fill up by resampling
    while len(sampled_example_bases) < num_examples:
        sampled_example_bases.append(random.choice(valid_base_filenames))

    # Ensure outlines are not from the example images
    remaining_bases_for_outlines = list(set(valid_base_filenames) - set(sampled_example_bases))
    if len(remaining_bases_for_outlines) < num_outlines_to_display:
        raise ValueError("Not enough unique files to ensure outlines are not from the example images.")

    num_unique_needed_for_outlines = min(num_outlines_to_display, len(remaining_bases_for_outlines))
    sampled_outline_bases = random.sample(remaining_bases_for_outlines, num_unique_needed_for_outlines)
    while len(sampled_outline_bases) < num_outlines_to_display:
        sampled_outline_bases.append(random.choice(remaining_bases_for_outlines))


    example_images_data = []
    for base in sampled_example_bases:
        edge_path = os.path.join(edge_dir, f"{base}_edgedetail.npy")
        img = np.load(edge_path).reshape(img_shape_tuple)
        example_images_data.append(img)

    outline_images_for_testing = []
    for base in sampled_outline_bases:
        outline_path = os.path.join(outline_dir, f"{base}_outline.npy")
        img = np.load(outline_path).reshape(img_shape_tuple)
        outline_images_for_testing.append(img)
        
    return example_images_data, outline_images_for_testing

# --- Prediction and Visualization ---
def predict_and_display(model, example_edge_imgs_list, input_outline_mask_list, threshold=0.5):
    num_outlines_to_show = len(input_outline_mask_list)
    
    batch_example_inputs = [np.expand_dims(img, axis=0) for img in example_edge_imgs_list]

    for i, outline_img_single in enumerate(input_outline_mask_list):
        batch_outline_input_for_generator = np.expand_dims(outline_img_single, axis=0)
        
        model_inputs_for_pred = batch_example_inputs + [batch_outline_input_for_generator]
        
        generated_batch = model.predict(model_inputs_for_pred)
        generated_single_outline = generated_batch[0]

        plt.figure(figsize=(18, 4)) # Adjusted figure size
        
        for j, ex_img in enumerate(example_edge_imgs_list):
            plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, j + 1)
            plt.imshow(ex_img.squeeze(), cmap='gray_r')
            plt.title(f"Ex Edge {j+1}")
            plt.axis('off')
            
        plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 1)
        plt.imshow(outline_img_single.squeeze(), cmap='gray_r')
        plt.title(f"Input Outline {i+1}")
        plt.axis('off')

        generated_display = (generated_single_outline.squeeze() > threshold).astype(float)
        plt.subplot(1, NUM_EXAMPLES_CONDITION + 2, NUM_EXAMPLES_CONDITION + 2)
        plt.imshow(generated_display, cmap='gray_r')
        plt.title(f"Generated Outline {i+1}")
        plt.axis('off')
        
        plt.suptitle(f"Prediction - Input Outline {i+1}")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

def main():
    num_test_sets = 3
    try:
        example_images, outline_images_for_test = get_prediction_samples(
            EDGE_DETAILS_DIR_SAMPLING, 
            OUTLINE_MASKS_DIR_SAMPLING,
            NUM_EXAMPLES_CONDITION,
            num_test_sets,
            IMG_SHAPE
        )
    except Exception as e:
        print(f"Error getting prediction samples: {e}")
        return

    print(f"Using {len(example_images)} example edge images and {len(outline_images_for_test)} input outline masks for prediction.")
    predict_and_display(trained_model, example_images, outline_images_for_test)

    # Optional: Test with a custom outline mask (must be a 128x128 .npy or .png file, 0 for shape, 1 for bg)
    # custom_outline_path = "path/to/your/custom_outline_mask.npy" 
    # if os.path.exists(custom_outline_path):
    #     if custom_outline_path.endswith(".npy"):
    #         custom_outline_np = np.load(custom_outline_path)
    #     # elif custom_outline_path.endswith(".png"): # You'd need a helper to load/process PNG to correct format
    #     #    from prepare_data import TARGET_SIZE # Example, adapt
    #     #    # Assuming a helper function to convert a new PNG to a B&W outline mask
    #     #    # custom_outline_np = process_new_png_to_outline(custom_outline_path, TARGET_SIZE)
    #     else:
    #         custom_outline_np = None
            
    #     if custom_outline_np is not None and custom_outline_np.shape == IMG_SHAPE[:2]:
    #         custom_outline_np = custom_outline_np.reshape(IMG_SHAPE)
    #         print("\nTesting with a custom outline mask:")
    #         # Use the same example_images loaded earlier
    #         predict_and_display(trained_model, example_images, [custom_outline_np])
    #     else:
    #         print(f"Could not process custom outline mask from {custom_outline_path} or shape mismatch.")


if __name__ == "__main__":
    main()