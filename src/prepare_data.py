# src/prepare_data.py
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import os
import glob
import random
import math
import traceback # Added for better error printing

TARGET_SIZE = (128, 128)
OUTPUT_DIR_BASE = "dataset/processed/grayscale_augmented_cat_v3" # Your provided path
OUTPUT_DIR_OUTLINES = os.path.join(OUTPUT_DIR_BASE, "outline_masks")
OUTPUT_DIR_GRAYSCALECATS = os.path.join(OUTPUT_DIR_BASE, "grayscale_cats")
INPUT_DIR_PNGS = "dataset/cat" # Your provided path

SCALING_PADDING = 4

# AUGMENTATION_FACTOR from your file, now representing content augmentations
AUGMENTATION_FACTOR = 10 # Base content + (AUGMENTATION_FACTOR-1) augmented content versions
N_APPROX_OUTLINES_PER_MAIN_OUTLINE = 3 # Generate N approximate outlines for each main outline (precise or geo-augmented)

# --- Augmentation Parameters (Same as your file) ---
ROTATION_RANGE = (-15, 15)
SCALE_RANGE = (0.85, 1.15)
TRANSLATION_RANGE_X = (-0.1, 0.1)
TRANSLATION_RANGE_Y = (-0.1, 0.1)
HORIZONTAL_FLIP_PROB = 0.5
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.7, 1.3)


def apply_geometric_augmentations_pil(img_pil, angle, scale, trans_x, trans_y, flip, is_mask_image=False): # Added is_mask_image
    """Applies geometric augmentations to a PIL Image."""
    fillcolor = None
    resample_method = Image.BICUBIC

    if is_mask_image: # Typically 'L' mode where object is 255, background is 0 in ROI
        fillcolor = 0 # Black for mask background within ROI
        resample_method = Image.NEAREST
    else: # Content image
        if img_pil.mode == 'RGBA':
            fillcolor = (255, 255, 255, 0) # Transparent white for RGBA content
        elif img_pil.mode == 'L':
            fillcolor = 255 # White for grayscale content that was already 'L'
        elif img_pil.mode == 'RGB':
             fillcolor = (255,255,255) # White for RGB
        else: # Default for other modes, or if unsure
            fillcolor = 255

    w, h = img_pil.size

    if flip:
        img_pil = ImageOps.mirror(img_pil)

    img_pil = img_pil.rotate(angle, resample=resample_method, expand=False, fillcolor=fillcolor)

    new_w_scaled, new_h_scaled = int(w * scale), int(h * scale)
    if new_w_scaled <= 0 or new_h_scaled <= 0: return img_pil # Avoid error with zero size
        
    scaled_img = img_pil.resize((new_w_scaled, new_h_scaled), resample=resample_method)
    
    paste_x = (w - new_w_scaled) // 2
    paste_y = (h - new_h_scaled) // 2
    
    new_img_for_scale = Image.new(img_pil.mode, (w,h), color=fillcolor)
    # For RGBA, using scaled_img as its own mask during paste preserves its transparency
    new_img_for_scale.paste(scaled_img, (paste_x, paste_y), scaled_img if img_pil.mode == 'RGBA' else None)
    img_pil = new_img_for_scale

    translate_pixels_x = int(trans_x * w)
    translate_pixels_y = int(trans_y * h)
    img_pil = img_pil.transform(img_pil.size, Image.AFFINE,
                                (1, 0, translate_pixels_x, 0, 1, translate_pixels_y),
                                resample=resample_method, fillcolor=fillcolor)
    return img_pil

# (apply_photometric_augmentations_pil remains the same as your provided file)
def apply_photometric_augmentations_pil(img_pil):
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(random.uniform(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]))
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(random.uniform(CONTRAST_RANGE[0], CONTRAST_RANGE[1]))
    return img_pil

def create_final_canvas_from_roi(roi_np, target_size_tuple, is_for_outline_mask_target=True):
    """
    Scales and centers an ROI onto a final canvas.
    roi_np: Input ROI.
            If for outline mask (is_for_outline_mask_target=True): expects object=255, bg=0.
            If for grayscale content: expects 0-1 float.
    is_for_outline_mask_target: If True, output canvas is 0.0 for object, 1.0 for bg.
                               If False, output canvas is 0.0-1.0 float, bg=1.0.
    """
    if roi_np is None or roi_np.size == 0: return None

    final_canvas = np.ones(target_size_tuple, dtype=np.float32) # Background is 1.0 (white)

    roi_h, roi_w = roi_np.shape[:2]
    if roi_w == 0 or roi_h == 0: return None

    scale_factor = min((target_size_tuple[0] - SCALING_PADDING) / roi_w, 
                       (target_size_tuple[1] - SCALING_PADDING) / roi_h)
    if scale_factor <= 0: return None # Should not happen if ROI has size
        
    new_w, new_h = max(1, int(roi_w * scale_factor)), max(1, int(roi_h * scale_factor))

    if is_for_outline_mask_target:
        # Ensure ROI is binary 0/255 before resize
        _, roi_binary_255 = cv2.threshold(roi_np.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        resized_roi = cv2.resize(roi_binary_255, (new_w, new_h), interpolation=cv2.INTER_NEAREST) # object is 255
    else: # Grayscale content (0-1 float)
        resized_roi = cv2.resize(roi_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

    start_x = (target_size_tuple[0] - new_w) // 2
    start_y = (target_size_tuple[1] - new_h) // 2
    
    if is_for_outline_mask_target:
        final_canvas[start_y:start_y+new_h, start_x:start_x+new_w][resized_roi == 255] = 0.0 # Object becomes 0.0 (black)
    else:
        # Place grayscale content. If it has transparent parts handled correctly (became white), this is fine.
        final_canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized_roi
    return final_canvas


def generate_approximate_outline_roi(base_precise_roi_mask_np, method_idx):
    """
    Generates one variation of an outline from a precise ROI mask (object=255, bg=0).
    Returns a modified ROI mask (object=255, bg=0).
    """
    h, w = base_precise_roi_mask_np.shape
    varied_mask_roi = base_precise_roi_mask_np.copy() # Work on a copy

    # Ensure input is uint8 for OpenCV functions
    if varied_mask_roi.dtype != np.uint8:
        varied_mask_roi = varied_mask_roi.astype(np.uint8)

    if method_idx == 0: # Morphological Closing
        k_size = random.choice([3, 5, 7])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        iterations = random.choice([1, 2])
        varied_mask_roi = cv2.morphologyEx(varied_mask_roi, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif method_idx == 1: # Morphological Opening
        k_size = random.choice([3, 5, 7])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        iterations = random.choice([1, 2])
        varied_mask_roi = cv2.morphologyEx(varied_mask_roi, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif method_idx == 2: # Contour Simplification
        contours, _ = cv2.findContours(varied_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0: # Avoid division by zero for tiny contours
                epsilon = random.uniform(0.005, 0.025) * perimeter 
                approx_poly = cv2.approxPolyDP(cnt, epsilon, True)
                temp_canvas = np.zeros_like(varied_mask_roi)
                cv2.drawContours(temp_canvas, [approx_poly], -1, 255, thickness=cv2.FILLED)
                varied_mask_roi = temp_canvas
    elif method_idx == 3: # Slight geometric distortions directly on the mask ROI
        pil_mask = Image.fromarray(varied_mask_roi, mode='L')
        angle = random.uniform(-7, 7) # Smaller range for approximation
        scale = random.uniform(0.97, 1.03) # Smaller range
        trans_x = random.uniform(-0.03, 0.03) # Smaller range
        trans_y = random.uniform(-0.03, 0.03) # Smaller range
        pil_mask_augmented = apply_geometric_augmentations_pil(pil_mask, angle, scale, trans_x, trans_y, False, is_mask_image=True)
        varied_mask_roi = np.array(pil_mask_augmented)
        _, varied_mask_roi = cv2.threshold(varied_mask_roi, 127, 255, cv2.THRESH_BINARY)

    if np.sum(varied_mask_roi > 0) < (h * w * 0.01): # Check if object part is too small
        return base_precise_roi_mask_np # Fallback
    return varied_mask_roi


def main():
    os.makedirs(OUTPUT_DIR_OUTLINES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_GRAYSCALECATS, exist_ok=True)

    image_paths = glob.glob(os.path.join(INPUT_DIR_PNGS, "*.png")) + \
                  glob.glob(os.path.join(INPUT_DIR_PNGS, "*.jpg")) + \
                  glob.glob(os.path.join(INPUT_DIR_PNGS, "*.jpeg"))
    if not image_paths:
        print(f"No image files found in {INPUT_DIR_PNGS}.")
        return

    print(f"Found {len(image_paths)} original images. AUGMENTATION_FACTOR (for content): {AUGMENTATION_FACTOR}, N_APPROX_OUTLINES_PER_MAIN_OUTLINE: {N_APPROX_OUTLINES_PER_MAIN_OUTLINE}")
    
    total_saved_pairs = 0
    for i, image_path in enumerate(image_paths):
        original_filename_base = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nProcessing original ({i+1}/{len(image_paths)}): {original_filename_base}")

        try:
            img_pil_orig_rgba = Image.open(image_path).convert('RGBA') # Ensure RGBA
            img_rgba_orig_np = np.array(img_pil_orig_rgba)

            alpha_channel_full_np = img_rgba_orig_np[:, :, 3]
            _, binary_mask_from_alpha_full_res_np = cv2.threshold(alpha_channel_full_np, 127, 255, cv2.THRESH_BINARY)

            if np.sum(binary_mask_from_alpha_full_res_np) == 0:
                print(f"  Skipping {original_filename_base}: No foreground in alpha.")
                continue

            contours, _ = cv2.findContours(binary_mask_from_alpha_full_res_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"  Skipping {original_filename_base}: No contours found.")
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(main_contour)

            # Base ROI mask (object=255, bg=0)
            base_precise_cat_roi_mask_np = np.zeros((h_bbox, w_bbox), dtype=np.uint8)
            shifted_contour = main_contour - [x_bbox, y_bbox]
            cv2.drawContours(base_precise_cat_roi_mask_np, [shifted_contour], -1, 255, thickness=cv2.FILLED)

            # Base RGBA content ROI
            base_cat_rgba_roi_np = img_rgba_orig_np[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox, :]
            
            # Loop for content augmentation (AUGMENTATION_FACTOR times)
            for aug_idx in range(AUGMENTATION_FACTOR):
                current_content_savename_base = f"{original_filename_base}"
                current_rgba_content_roi_pil = Image.fromarray(base_cat_rgba_roi_np, 'RGBA')
                current_precise_mask_roi_pil = Image.fromarray(base_precise_cat_roi_mask_np, 'L') # object=255, bg=0

                if aug_idx > 0: # Apply geometric and photometric augmentations for content
                    current_content_savename_base += f"_aug{aug_idx}"
                    # print(f"  Generating augmented content version {aug_idx} for {original_filename_base}")
                    geo_aug_params = {
                        'angle': random.uniform(ROTATION_RANGE[0], ROTATION_RANGE[1]),
                        'scale': random.uniform(SCALE_RANGE[0], SCALE_RANGE[1]),
                        'trans_x': random.uniform(TRANSLATION_RANGE_X[0], TRANSLATION_RANGE_X[1]),
                        'trans_y': random.uniform(TRANSLATION_RANGE_Y[0], TRANSLATION_RANGE_Y[1]),
                        'flip': random.random() < HORIZONTAL_FLIP_PROB
                    }
                    # Augment content (RGBA)
                    current_rgba_content_roi_pil = apply_geometric_augmentations_pil(
                        current_rgba_content_roi_pil, **geo_aug_params, is_mask_image=False)
                    # Augment its corresponding precise mask with THE SAME geometric params
                    current_precise_mask_roi_pil = apply_geometric_augmentations_pil(
                        current_precise_mask_roi_pil, **geo_aug_params, is_mask_image=True)
                
                # Convert content to grayscale and apply photometric augmentation (if aug_idx > 0)
                current_gray_content_roi_pil = current_rgba_content_roi_pil.convert('L')
                if aug_idx > 0:
                    current_gray_content_roi_pil = apply_photometric_augmentations_pil(current_gray_content_roi_pil)
                
                # Final Grayscale Content ROI (0-1 float for saving)
                final_gray_content_roi_np = np.array(current_gray_content_roi_pil).astype(np.float32) / 255.0
                
                # The precise mask ROI for the current content version (0/255 numpy)
                final_precise_mask_roi_for_current_content_np = np.array(current_precise_mask_roi_pil)
                _, final_precise_mask_roi_for_current_content_np = cv2.threshold( # Ensure binary
                    final_precise_mask_roi_for_current_content_np, 127, 255, cv2.THRESH_BINARY)

                # Create final canvas for this grayscale content
                # Needs to handle background correctly based on its own mask
                temp_grayscale_canvas_roi = np.ones_like(final_gray_content_roi_np, dtype=np.float32)
                temp_grayscale_canvas_roi[final_precise_mask_roi_for_current_content_np == 255] = \
                    final_gray_content_roi_np[final_precise_mask_roi_for_current_content_np == 255]

                final_grayscale_canvas = create_final_canvas_from_roi(
                    temp_grayscale_canvas_roi, TARGET_SIZE, is_for_outline_mask_target=False)

                if final_grayscale_canvas is None: 
                    print(f"    Skipping content {current_content_savename_base}: Grayscale canvas failed.")
                    continue
                
                # 1. Save the (precise outline, corresponding grayscale content) pair
                # This precise outline corresponds to the geometry of final_grayscale_canvas
                final_precise_outline_canvas = create_final_canvas_from_roi(
                    final_precise_mask_roi_for_current_content_np.copy(), TARGET_SIZE, is_for_outline_mask_target=True)

                if final_precise_outline_canvas is not None:
                    gs_path = os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{current_content_savename_base}_grayscale.npy")
                    ol_path = os.path.join(OUTPUT_DIR_OUTLINES, f"{current_content_savename_base}_outline.npy")
                    np.save(gs_path, final_grayscale_canvas)
                    np.save(ol_path, final_precise_outline_canvas)
                    total_saved_pairs += 1
                else:
                    print(f"    Skipping content {current_content_savename_base}: Precise outline canvas failed.")
                    continue # Skip approx outlines if precise one failed for this content

                # 2. Generate N_APPROX_OUTLINES_PER_MAIN_OUTLINE for the current content's shape
                for approx_jdx in range(N_APPROX_OUTLINES_PER_MAIN_OUTLINE):
                    approx_outline_savename_base = f"{current_content_savename_base}_approx_{approx_jdx}"
                    
                    # Approximation is done on the ROI mask that defines the current content's shape
                    approximated_roi_mask_np = generate_approximate_outline_roi(
                        final_precise_mask_roi_for_current_content_np.copy(), 
                        method_idx = 2 # using method 2 only as it seems best
                        # approx_jdx % 4 # Cycle through approximation methods
                    )
                    if approximated_roi_mask_np is None: continue

                    final_approx_outline_canvas = create_final_canvas_from_roi(
                        approximated_roi_mask_np, TARGET_SIZE, is_for_outline_mask_target=True)

                    if final_approx_outline_canvas is not None:
                        # Save the approximate outline
                        approx_ol_path = os.path.join(OUTPUT_DIR_OUTLINES, f"{approx_outline_savename_base}_outline.npy")
                        np.save(approx_ol_path, final_approx_outline_canvas)
                        
                        # preview
                        # cv2.imwrite(approx_ol_path.replace(".npy", "_preview.png"), (final_approx_outline_canvas * 255).astype(np.uint8))
                        
                        # Save a corresponding grayscale file (which is a copy of the current content's grayscale)
                        approx_gs_path = os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{approx_outline_savename_base}_grayscale.npy")
                        np.save(approx_gs_path, final_grayscale_canvas) # Save the same grayscale content
                        total_saved_pairs += 1
                    else:
                        print(f"    Skipping approx outline {approx_jdx} for {current_content_savename_base}: Approx outline canvas failed.")

                # Optional Preview for the first original image's base and first aug content
                if i == 0 and (aug_idx == 0 or aug_idx == 1):
                    if final_grayscale_canvas is not None:
                        cv2.imwrite(os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{current_content_savename_base}_grayscale_preview.png"), (final_grayscale_canvas * 255).astype(np.uint8))
                    if final_precise_outline_canvas is not None:
                        cv2.imwrite(os.path.join(OUTPUT_DIR_OUTLINES, f"{current_content_savename_base}_outline_preview.png"), (final_precise_outline_canvas * 255).astype(np.uint8))


        except Exception as e:
            print(f"!! Unhandled error processing {original_filename_base} (content aug_idx {aug_idx if 'aug_idx' in locals() else 'N/A'}): {e}")
            traceback.print_exc()
            continue

    print(f"\nFinished processing. Total saved (grayscale, outline) pairs: {total_saved_pairs}.")
    print(f"Outline masks saved in: {OUTPUT_DIR_OUTLINES}")
    print(f"Grayscale cat images saved in: {OUTPUT_DIR_GRAYSCALECATS}")

if __name__ == "__main__":
    main()