# src/prepare_data.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os
import glob
import random
import math

TARGET_SIZE = (128, 128) # Or (64, 64) if you revert
OUTPUT_DIR_BASE = "dataset/processed/grayscale_augmented_cat" # Preserved path
OUTPUT_DIR_OUTLINES = os.path.join(OUTPUT_DIR_BASE, "outline_masks")
OUTPUT_DIR_GRAYSCALECATS = os.path.join(OUTPUT_DIR_BASE, "grayscale_cats") # New folder
INPUT_DIR_PNGS = "dataset/cat" # Preserved path

SCALING_PADDING = 4 # Pixels of padding when scaling to fit TARGET_SIZE

AUGMENTATION_FACTOR = 10 # Generate original + (AUGMENTATION_FACTOR-1) augmented versions for each image.
                         # Set to 1 for no offline augmentation beyond the base image.

# --- Augmentation Parameters ---
# Geometric
ROTATION_RANGE = (-15, 15) # degrees
SCALE_RANGE = (0.85, 1.15) # zoom factor
TRANSLATION_RANGE_X = (-0.1, 0.1) # fraction of width
TRANSLATION_RANGE_Y = (-0.1, 0.1) # fraction of height
HORIZONTAL_FLIP_PROB = 0.5

# Photometric (applied only to grayscale cat image)
BRIGHTNESS_RANGE = (0.7, 1.3) # Multiplicative factor for PIL.ImageEnhance.Brightness
CONTRAST_RANGE = (0.7, 1.3)   # Multiplicative factor for PIL.ImageEnhance.Contrast


def apply_geometric_augmentations_pil(img_pil, angle, scale, trans_x, trans_y, flip, is_mask_image=False):
    """
    Applies geometric augmentations to a PIL Image.
    is_mask_image: If True, uses NEAREST resampling and appropriate fill for binary masks.
    """
    fillcolor = None
    resample_method = Image.BICUBIC

    if is_mask_image: # Typically 'L' mode where object is 255, background is 0 in ROI
        fillcolor = 0 # Black for mask background within ROI
        resample_method = Image.NEAREST
    else: # Content image, typically RGBA at this stage
        if img_pil.mode == 'RGBA':
            fillcolor = (255, 255, 255, 0) # Transparent white for RGBA content
        elif img_pil.mode == 'L': # Grayscale content (less common at this direct call)
            fillcolor = 255 # White
        elif img_pil.mode == 'RGB':
            fillcolor = (255,255,255) # White
        else: # Default
            fillcolor = 255


    w, h = img_pil.size

    # 1. Flip
    if flip:
        img_pil = ImageOps.mirror(img_pil)

    # 2. Rotate (around center)
    img_pil = img_pil.rotate(angle, resample=resample_method, expand=False, fillcolor=fillcolor)

    # 3. Scale (around center)
    new_w_scaled, new_h_scaled = int(w * scale), int(h * scale)
    if new_w_scaled <= 0 or new_h_scaled <= 0: # Prevent zero-size image
        return img_pil # Or handle error appropriately
        
    scaled_img = img_pil.resize((new_w_scaled, new_h_scaled), resample=resample_method)
    
    # Create a new image with appropriate background and paste the scaled image in the center
    paste_x = (w - new_w_scaled) // 2
    paste_y = (h - new_h_scaled) // 2
    
    # Create new_img_for_scale with the same fillcolor used for transformations
    new_img_for_scale = Image.new(img_pil.mode, (w,h), color=fillcolor)
    new_img_for_scale.paste(scaled_img, (paste_x, paste_y)) # Pasting RGBA with alpha respects transparency
    img_pil = new_img_for_scale


    # 4. Translate
    translate_pixels_x = int(trans_x * w)
    translate_pixels_y = int(trans_y * h)
    img_pil = img_pil.transform(img_pil.size, Image.AFFINE,
                                (1, 0, translate_pixels_x, 0, 1, translate_pixels_y),
                                resample=resample_method, fillcolor=fillcolor)
    return img_pil


def apply_photometric_augmentations_pil(img_pil):
    """Applies photometric augmentations to a PIL Image (assumed grayscale 'L' mode)."""
    # Brightness
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]))
    # Contrast
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1]))
    return img_pil


def create_filled_outline_mask(precise_cat_roi_mask_orig_res_np, target_size_tuple,
                               augmentation_params=None): # Removed is_mask, implied by function
    """
    Creates the filled B&W outline mask, optionally augmented.
    Mask values: 0 for cat object, 1 for background. Background is white
    precise_cat_roi_mask_orig_res_np: object is 255, background is 0
    """
    if precise_cat_roi_mask_orig_res_np is None or precise_cat_roi_mask_orig_res_np.size == 0:
        return None

    img_pil = Image.fromarray(precise_cat_roi_mask_orig_res_np, mode='L') # Mask: object 255, bg 0

    if augmentation_params:
        img_pil = apply_geometric_augmentations_pil(
            img_pil,
            augmentation_params['angle'],
            augmentation_params['scale'],
            augmentation_params['trans_x'],
            augmentation_params['trans_y'],
            augmentation_params['flip'],
            is_mask_image=True # Crucial: treat as mask
        )

    augmented_roi_mask_np = np.array(img_pil)
    _, augmented_roi_mask_np_binarized = cv2.threshold(augmented_roi_mask_np, 127, 255, cv2.THRESH_BINARY) # Object is 255

    outline_image_final = np.ones(target_size_tuple, dtype=np.float32) # Target: background 1 (white)
    
    roi_h, roi_w = augmented_roi_mask_np_binarized.shape
    if roi_w == 0 or roi_h == 0: return None

    scale_factor = min((target_size_tuple[0] - SCALING_PADDING) / roi_w, (target_size_tuple[1] - SCALING_PADDING) / roi_h)
    if scale_factor <= 0: return None
        
    new_w, new_h = max(1, int(roi_w * scale_factor)), max(1, int(roi_h * scale_factor))
    resized_cat_roi_mask = cv2.resize(augmented_roi_mask_np_binarized, (new_w, new_h), interpolation=cv2.INTER_NEAREST) # Object is 255

    start_x = (target_size_tuple[0] - new_w) // 2
    start_y = (target_size_tuple[1] - new_h) // 2
    
    outline_image_final[start_y:start_y+new_h, start_x:start_x+new_w][resized_cat_roi_mask == 255] = 0.0 # Object becomes 0 (black)
    return outline_image_final


def create_grayscale_cat_image(cat_color_roi_orig_res_np_rgba, precise_cat_roi_mask_orig_res_np, target_size_tuple,
                               augmentation_params=None):
    """
    Creates a grayscale cat image, optionally augmented, scaled and centered.
    cat_color_roi_orig_res_np_rgba: RGBA numpy array for content.
    precise_cat_roi_mask_orig_res_np: Binary mask (0/255) for shape.
    """
    if cat_color_roi_orig_res_np_rgba is None or precise_cat_roi_mask_orig_res_np is None:
        return None
    
    # Convert numpy arrays to PIL Images
    img_pil_rgba_content = Image.fromarray(cat_color_roi_orig_res_np_rgba, mode='RGBA')
    img_pil_shape_mask = Image.fromarray(precise_cat_roi_mask_orig_res_np, mode='L') # Mask: object 255, bg 0

    current_geometrically_augmented_mask_np = precise_cat_roi_mask_orig_res_np # Default if no aug

    if augmentation_params:
        # Apply geometric augmentations to the RGBA color content
        img_pil_rgba_content = apply_geometric_augmentations_pil(
            img_pil_rgba_content,
            augmentation_params['angle'],
            augmentation_params['scale'],
            augmentation_params['trans_x'],
            augmentation_params['trans_y'],
            augmentation_params['flip'],
            is_mask_image=False # This is content
        )
        # Apply THE SAME geometric augmentations to the shape mask
        img_pil_shape_mask = apply_geometric_augmentations_pil(
            img_pil_shape_mask, # This is the L mode mask
            augmentation_params['angle'],
            augmentation_params['scale'],
            augmentation_params['trans_x'],
            augmentation_params['trans_y'],
            augmentation_params['flip'],
            is_mask_image=True # This is a mask
        )
        # Binarize the geometrically augmented shape mask
        temp_mask_np = np.array(img_pil_shape_mask)
        _, current_geometrically_augmented_mask_np = cv2.threshold(temp_mask_np, 127, 255, cv2.THRESH_BINARY)


    # Convert (geometrically augmented) RGBA content to Grayscale ('L' mode)
    # Transparent areas (filled with (255,255,255,0) during aug) will become white (255)
    img_pil_gray_content = img_pil_rgba_content.convert('L')

    if augmentation_params: # Apply photometric to the (geometrically augmented) grayscale content
        img_pil_gray_content = apply_photometric_augmentations_pil(img_pil_gray_content)

    augmented_gray_cat_content_np = np.array(img_pil_gray_content) # 0-255 range
    normalized_gray_content = augmented_gray_cat_content_np.astype(np.float32) / 255.0 # 0-1 range
    
    # Final canvas (white background 1.0)
    grayscale_image_final = np.ones(target_size_tuple, dtype=np.float32)

    # Scaling and centering logic using the (geometrically augmented) shape mask
    roi_h, roi_w = current_geometrically_augmented_mask_np.shape # Mask is 0/255
    if roi_w == 0 or roi_h == 0: return None

    scale_factor = min((target_size_tuple[0] - SCALING_PADDING) / roi_w, (target_size_tuple[1] - SCALING_PADDING) / roi_h)
    if scale_factor <= 0: return None
        
    new_w, new_h = max(1, int(roi_w * scale_factor)), max(1, int(roi_h * scale_factor))

    # Resize the normalized grayscale cat content
    resized_gray_cat_content = cv2.resize(normalized_gray_content, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Resize the corresponding (geometrically augmented and binarized) shape mask
    resized_final_shape_mask = cv2.resize(current_geometrically_augmented_mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST) # Mask is 0/255
    resized_final_shape_mask_bool = (resized_final_shape_mask == 255) # Boolean: True where object is

    start_x = (target_size_tuple[0] - new_w) // 2
    start_y = (target_size_tuple[1] - new_h) // 2

    canvas_roi = grayscale_image_final[start_y:start_y+new_h, start_x:start_x+new_w]
    # Place the grayscale content onto the white canvas ONLY where the final shape mask is True
    canvas_roi[resized_final_shape_mask_bool] = resized_gray_cat_content[resized_final_shape_mask_bool]
    grayscale_image_final[start_y:start_y+new_h, start_x:start_x+new_w] = canvas_roi
    
    return grayscale_image_final


def main():
    os.makedirs(OUTPUT_DIR_OUTLINES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_GRAYSCALECATS, exist_ok=True)

    image_paths = glob.glob(os.path.join(INPUT_DIR_PNGS, "*.png")) + \
                  glob.glob(os.path.join(INPUT_DIR_PNGS, "*.jpg")) + \
                  glob.glob(os.path.join(INPUT_DIR_PNGS, "*.jpeg"))
    if not image_paths:
        print(f"No image files (png, jpg, jpeg) found in {INPUT_DIR_PNGS}. Please check the path.")
        return

    print(f"Found {len(image_paths)} original image files. Starting processing to {TARGET_SIZE}...")
    if AUGMENTATION_FACTOR > 1:
        print(f"Will generate {AUGMENTATION_FACTOR-1} augmented versions for each image.")
    
    total_files_generated = 0
    for i, image_path in enumerate(image_paths):
        original_filename_base = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing original ({i+1}/{len(image_paths)}): {original_filename_base}")

        try:
            img_pil_orig = Image.open(image_path)
            if img_pil_orig.mode != 'RGBA': # Ensure RGBA for consistent alpha handling
                img_pil_orig = img_pil_orig.convert('RGBA')
            img_rgba_orig_np = np.array(img_pil_orig)

            alpha_channel_full_np = img_rgba_orig_np[:, :, 3]
            _, binary_mask_from_alpha_full_res_np = cv2.threshold(alpha_channel_full_np, 127, 255, cv2.THRESH_BINARY)

            if np.sum(binary_mask_from_alpha_full_res_np) == 0:
                print(f"  Skipping {original_filename_base}: No foreground in alpha channel.")
                continue

            contours, _ = cv2.findContours(binary_mask_from_alpha_full_res_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"  Skipping {original_filename_base}: No contours found from alpha mask.")
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(main_contour)

            # Mask of the cat *within its bounding box* (object=255, background=0)
            precise_cat_roi_mask_orig_res_np = np.zeros((h_bbox, w_bbox), dtype=np.uint8)
            shifted_contour = main_contour - [x_bbox, y_bbox]
            cv2.drawContours(precise_cat_roi_mask_orig_res_np, [shifted_contour], -1, color=255, thickness=cv2.FILLED)

            # RGBA content of the cat *within its bounding box*
            cat_rgba_roi_orig_res_np = img_rgba_orig_np[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox, :]
            
            for aug_idx in range(AUGMENTATION_FACTOR):
                current_filename_base = f"{original_filename_base}"
                aug_params = None

                if aug_idx > 0: 
                    current_filename_base += f"_aug{aug_idx}"
                    aug_params = {
                        'angle': random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1]),
                        'scale': random.uniform(SCALE_RANGE[0], SCALE_RANGE[1]),
                        'trans_x': random.uniform(TRANSLATION_RANGE_X[0], TRANSLATION_RANGE_X[1]),
                        'trans_y': random.uniform(TRANSLATION_RANGE_Y[0], TRANSLATION_RANGE_Y[1]),
                        'flip': random.random() < HORIZONTAL_FLIP_PROB
                    }
                    # print(f"  Generating augmented version {aug_idx}: {current_filename_base}")


                outline_mask_img = create_filled_outline_mask(
                    precise_cat_roi_mask_orig_res_np.copy(), # This is object=255, bg=0 ROI mask
                    TARGET_SIZE, 
                    aug_params
                )
                if outline_mask_img is not None:
                    save_path_outline = os.path.join(OUTPUT_DIR_OUTLINES, f"{current_filename_base}_outline.npy")
                    np.save(save_path_outline, outline_mask_img)
                else:
                    print(f"  Failed to create outline mask for {current_filename_base}. Skipping this version.")
                    continue
                
                grayscale_cat_img = create_grayscale_cat_image(
                    cat_rgba_roi_orig_res_np.copy(), # This is RGBA content ROI
                    precise_cat_roi_mask_orig_res_np.copy(), # This is object=255, bg=0 shape ROI mask
                    TARGET_SIZE, 
                    aug_params
                )
                if grayscale_cat_img is not None:
                    save_path_grayscale = os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{current_filename_base}_grayscale.npy")
                    np.save(save_path_grayscale, grayscale_cat_img)
                    
                    # Optional preview for the first image's augmentations, or all base images
                    if i == 0 and aug_idx < 5 : # Preview some augmentations of the first image
                       cv2.imwrite(os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{current_filename_base}_grayscale_preview.png"), (grayscale_cat_img * 255).astype(np.uint8))
                       cv2.imwrite(os.path.join(OUTPUT_DIR_OUTLINES, f"{current_filename_base}_outline_preview.png"), (outline_mask_img * 255).astype(np.uint8))
                    elif aug_idx == 0 and i > 0: # Preview base image for other images
                       cv2.imwrite(os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{current_filename_base}_grayscale_preview.png"), (grayscale_cat_img * 255).astype(np.uint8))
                       cv2.imwrite(os.path.join(OUTPUT_DIR_OUTLINES, f"{current_filename_base}_outline_preview.png"), (outline_mask_img * 255).astype(np.uint8))


                else:
                    print(f"  Failed to create grayscale cat image for {current_filename_base}. Skipping this version.")
                    if os.path.exists(save_path_outline): os.remove(save_path_outline)
                    continue
                
                total_files_generated +=1

        except Exception as e:
            print(f"  Unhandled error processing {original_filename_base} (or its augmentations): {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nFinished processing. Successfully created files for {total_files_generated} image versions (base + augmentations).")
    print(f"Outline masks saved in: {OUTPUT_DIR_OUTLINES}")
    print(f"Grayscale cat images saved in: {OUTPUT_DIR_GRAYSCALECATS}")

if __name__ == "__main__":
    main()