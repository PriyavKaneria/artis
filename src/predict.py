# src/predict.py
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt # Not strictly needed for just printing tokens
import os
import random
import glob
# import cv2 # Not strictly needed for this specific test

# --- Import custom layers and model builder ---
from model_def import build_combined_model_phase1, PatchEmbed, StyleEncoder, FiLMLayer # FiLMLayer for completeness if model saved with it

# --- Configuration ---
IMG_DIM = 128
IMG_SHAPE = (IMG_DIM, IMG_DIM, 1)
NUM_EXAMPLES_CONDITION = 5 # Must match how the StyleEncoder was trained

# Model path for the combined model trained in Phase 1
MODEL_PATH = "trained_models/cat_generator_style_encoder_v1.keras" # Or .h5 if you saved that

# Data paths (using your existing structure for base grayscale images)
OUTPUT_DIR_BASE_PROCESSED = "dataset/processed/grayscale_augmented_cat_v3/"
GRAYSCALE_CATS_DIR_SAMPLING = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "grayscale_cats/")
# OUTLINE_MASKS_DIR_SAMPLING = os.path.join(OUTPUT_DIR_BASE_PROCESSED, "outline_masks/") # Not needed for this test

# StyleEncoder HParams (must match how the loaded model was built)
PATCH_SIZE_STYLE_ENC = 16
EMBED_DIM_STYLE_ENC = 128
NUM_HEADS_STYLE_ENC = 4
NUM_TRANS_LAYERS_STYLE_ENC = 2
NUM_STYLE_TOKENS_ENC = 5

print(f"Sampling grayscale from: {GRAYSCALE_CATS_DIR_SAMPLING}")

# --- Load Full Combined Model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}.")
try:
    # Provide all custom objects the model might contain
    custom_objects_dict = {
        'FiLMLayer': FiLMLayer,
        'PatchEmbed': PatchEmbed,
        'StyleEncoder': StyleEncoder,
        # IMPORTANT: If LeakyReLU was used as a string 'leaky_relu' before, 
        # but now it's a layer, this is fine.
        # The problem was Keras trying to deserialize the *layer instance* from within the Conv2D config.
    }
    full_trained_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects_dict)
    print("Full trained model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure custom objects are correctly registered.")
    exit()

# --- Extract or Rebuild the StyleEncoder part ---
# Option 1: If StyleEncoder was named and is a direct layer of the combined model
try:
    style_encoder_layer = full_trained_model.get_layer('style_encoder') # Must match the name in build_combined_model_phase1
    print("StyleEncoder layer extracted from the combined model.")
    # We can directly call this layer if its input signature matches what we provide.
    # It expects a list of 5 input tensors.
except ValueError:
    print("Could not find a layer named 'style_encoder'. Rebuilding and transferring weights if possible, or using direct call if full model has suitable inputs.")
    # Option 2: Rebuild the StyleEncoder with the same config and set weights (more complex)
    # This is harder if the StyleEncoder class itself wasn't saved as a "model" with its own weights.
    # For now, let's assume we can use the full model's inputs to get the style_tokens_output if direct layer access fails.
    # Or, the StyleEncoder model object itself was what we wanted.
    # In our build_combined_model_phase1, style_encoder_model IS the StyleEncoder instance.
    # We need a way to make style_encoder_model callable with new inputs if we don't want to predict through the whole graph.
    
    # Simplest for now: Create an intermediate model from the full model's inputs to the style_tokens_output tensor.
    # Find the tensor that is the output of style_encoder_model.call() before it's flattened.
    # This requires knowing the name of that tensor or finding the layer.
    # Let's assume `style_encoder_layer` above worked if StyleEncoder was named.
    # If not, we can try to get it by finding a layer of type StyleEncoder:
    found_style_encoder = None
    for layer in full_trained_model.layers:
        if isinstance(layer, StyleEncoder):
            found_style_encoder = layer
            break
    if found_style_encoder:
        style_encoder_layer = found_style_encoder
        print("StyleEncoder instance found within the combined model's layers.")
    else:
        print("Could not extract StyleEncoder. Exiting. Check model structure and layer naming.")
        exit()


# --- Helper to load base grayscale image paths ---
def get_base_grayscale_paths(grayscale_dir):
    gs_pattern = os.path.join(grayscale_dir, "*_grayscale.npy")
    all_gs_paths = sorted(glob.glob(gs_pattern))
    base_gs_paths = [p for p in all_gs_paths if "_aug" not in p and "_approx" not in p]
    return base_gs_paths

BASE_GRAYSCALE_PATHS = get_base_grayscale_paths(GRAYSCALE_CATS_DIR_SAMPLING)
if not BASE_GRAYSCALE_PATHS:
    print("CRITICAL: No base grayscale images found for examples. Exiting.")
    exit()
if len(BASE_GRAYSCALE_PATHS) < NUM_EXAMPLES_CONDITION:
    print(f"Warning: Only {len(BASE_GRAYSCALE_PATHS)} base images available, less than {NUM_EXAMPLES_CONDITION} needed for examples.")


# --- Prediction and Verification ---
def test_style_encoder(style_encoder_model_or_layer, num_test_sets=3):
    print(f"\n--- Testing Style Encoder with {num_test_sets} random example sets ---")

    for i in range(num_test_sets):
        print(f"\nTest Set {i+1}:")
        
        # Prepare 5 random example images (from base grayscales)
        num_available_examples = len(BASE_GRAYSCALE_PATHS)
        if num_available_examples < NUM_EXAMPLES_CONDITION:
            selected_example_paths = random.choices(BASE_GRAYSCALE_PATHS, k=NUM_EXAMPLES_CONDITION) # Sample with replacement
        else:
            selected_example_paths = random.sample(BASE_GRAYSCALE_PATHS, NUM_EXAMPLES_CONDITION)
        
        example_images_np_list = []
        print("  Example image files used:")
        for p_idx, path in enumerate(selected_example_paths):
            print(f"    Ex {p_idx+1}: {os.path.basename(path)}")
            try:
                img = np.load(path).reshape(IMG_SHAPE)
                example_images_np_list.append(img)
            except Exception as e:
                print(f"    Error loading {path}: {e}")
                return # Stop if an image fails to load

        if len(example_images_np_list) != NUM_EXAMPLES_CONDITION:
            print(f"  Could not load {NUM_EXAMPLES_CONDITION} examples for test set {i+1}. Skipping.")
            continue

        # The StyleEncoder model expects a list of tensors, each [B, H, W, C]
        # For prediction with a single set (B=1):
        example_tensors_for_style_encoder = [tf.expand_dims(tf.convert_to_tensor(img_np, dtype=tf.float32), axis=0) 
                                             for img_np in example_images_np_list]

        # Get style tokens
        try:
            # If style_encoder_layer is the actual StyleEncoder Model instance
            output_style_tokens = style_encoder_model_or_layer(example_tensors_for_style_encoder)
        except Exception as e:
            print(f"  Error calling StyleEncoder: {e}")
            print("  This might happen if style_encoder_model_or_layer is not the correct callable Keras Model/Layer.")
            print("  Ensure the StyleEncoder sub-model was correctly extracted or built.")
            continue


        print(f"  Output Style Tokens Shape: {output_style_tokens.shape}") # Expected: (1, num_style_tokens, embed_dim_style)
        
        # Print some statistics of the tokens
        # Flatten tokens for easier stats: (1, num_style_tokens * embed_dim_style)
        tokens_flat = tf.reshape(output_style_tokens, [1, -1]).numpy().squeeze()
        
        print(f"  Style Tokens (first few values of flattened tokens for this set): {tokens_flat[:10]}")
        print(f"  Mean of tokens: {np.mean(tokens_flat):.4f}, Std: {np.std(tokens_flat):.4f}, Min: {np.min(tokens_flat):.4f}, Max: {np.max(tokens_flat):.4f}")

        # For more advanced: calculate cosine similarity between token sets from different inputs
        # For now, just observe if values change with different inputs.

# --- Main Execution ---
def main():
    if style_encoder_layer: # Check if we successfully got the layer
        test_style_encoder(style_encoder_layer, num_test_sets=5)
    else:
        print("StyleEncoder could not be prepared for testing.")

if __name__ == "__main__":
    main()