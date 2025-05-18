# src/prepare_data.py
import cv2
import numpy as np
from PIL import Image
import os
import glob

TARGET_SIZE = (128, 128) # Or (64, 64) if you revert
OUTPUT_DIR_BASE = "dataset/processed/grayscale_cat" # Preserved path
OUTPUT_DIR_OUTLINES = os.path.join(OUTPUT_DIR_BASE, "outline_masks")
OUTPUT_DIR_GRAYSCALECATS = os.path.join(OUTPUT_DIR_BASE, "grayscale_cats") # New folder
INPUT_DIR_PNGS = "dataset/cat" # Preserved path

SCALING_PADDING = 4 # Pixels of padding when scaling to fit TARGET_SIZE

def create_filled_outline_mask(precise_cat_roi_mask_orig_res, target_size):
    """
    Creates the filled B&W outline mask. Cat is 0 (black), background is 1 (white).
    precise_cat_roi_mask_orig_res: The cat's shape (255 for cat, 0 for bg) cropped to its bounding box from original resolution.
    """
    if precise_cat_roi_mask_orig_res is None or precise_cat_roi_mask_orig_res.size == 0:
        return None

    # Final canvas: background white (1.0), cat will be black (0.0)
    outline_image_final = np.ones(target_size, dtype=np.float32)
    
    roi_h, roi_w = precise_cat_roi_mask_orig_res.shape
    if roi_w == 0 or roi_h == 0: return None

    scale = min((target_size[0] - SCALING_PADDING) / roi_w, (target_size[1] - SCALING_PADDING) / roi_h)
    if scale <= 0: return None
        
    new_w, new_h = max(1, int(roi_w * scale)), max(1, int(roi_h * scale))

    # Resize the cat's ROI mask (which is 0 or 255)
    resized_cat_roi_mask = cv2.resize(precise_cat_roi_mask_orig_res, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    start_x = (target_size[0] - new_w) // 2
    start_y = (target_size[1] - new_h) // 2

    # Where resized_cat_roi_mask is 255 (cat), make it 0.0 (black) in the final image
    outline_image_final[start_y:start_y+new_h, start_x:start_x+new_w][resized_cat_roi_mask == 255] = 0.0
    return outline_image_final


def create_grayscale_cat_image(cat_color_roi_orig_res, precise_cat_roi_mask_orig_res, target_size):
    """
    Creates a grayscale cat image, scaled and centered on a white background.
    Values are normalized between 0.0 (black) and 1.0 (white).
    cat_color_roi_orig_res: The RGB/RGBA part of the cat, cropped to its bounding box.
    precise_cat_roi_mask_orig_res: The cat's precise shape (0 or 255) within that bounding box.
    """
    if cat_color_roi_orig_res is None or precise_cat_roi_mask_orig_res is None:
        return None
    
    # 1. Convert color ROI to grayscale
    if cat_color_roi_orig_res.shape[2] == 4: # RGBA
        gray_cat_color_roi = cv2.cvtColor(cat_color_roi_orig_res, cv2.COLOR_RGBA2GRAY)
    else: # RGB
        gray_cat_color_roi = cv2.cvtColor(cat_color_roi_orig_res, cv2.COLOR_RGB2GRAY)

    # Normalize grayscale ROI to 0-1 float
    gray_cat_color_roi_normalized = gray_cat_color_roi.astype(np.float32) / 255.0
    
    # 2. Prepare final canvas (white background 1.0)
    grayscale_image_final = np.ones(target_size, dtype=np.float32)

    # 3. Scale the grayscale ROI and its mask to fit target_size
    roi_h, roi_w = precise_cat_roi_mask_orig_res.shape
    if roi_w == 0 or roi_h == 0: return None

    scale = min((target_size[0] - SCALING_PADDING) / roi_w, (target_size[1] - SCALING_PADDING) / roi_h)
    if scale <= 0: return None
        
    new_w, new_h = max(1, int(roi_w * scale)), max(1, int(roi_h * scale))

    # Resize the normalized grayscale cat content
    resized_gray_cat_content = cv2.resize(gray_cat_color_roi_normalized, (new_w, new_h), interpolation=cv2.INTER_AREA) # INTER_AREA for downscaling
    # Resize the mask (0 or 255)
    resized_alpha_mask = cv2.resize(precise_cat_roi_mask_orig_res, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    # Convert mask to boolean or 0/1 float for multiplication
    resized_alpha_mask_bool = (resized_alpha_mask == 255)


    start_x = (target_size[0] - new_w) // 2
    start_y = (target_size[1] - new_h) // 2

    # Place the scaled grayscale cat onto the white canvas using the alpha mask
    # Get the region on the canvas
    canvas_roi = grayscale_image_final[start_y:start_y+new_h, start_x:start_x+new_w]
    # Update only where the mask is true
    canvas_roi[resized_alpha_mask_bool] = resized_gray_cat_content[resized_alpha_mask_bool]
    grayscale_image_final[start_y:start_y+new_h, start_x:start_x+new_w] = canvas_roi
    
    return grayscale_image_final


def main():
    os.makedirs(OUTPUT_DIR_OUTLINES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_GRAYSCALECATS, exist_ok=True) # Create new folder

    image_paths = glob.glob(os.path.join(INPUT_DIR_PNGS, "*.png"))
    if not image_paths:
        print(f"No PNG files found in {INPUT_DIR_PNGS}. Please check the path.")
        return

    print(f"Found {len(image_paths)} PNG files. Starting processing to {TARGET_SIZE}...")
    
    processed_count = 0
    for i, image_path in enumerate(image_paths):
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing ({i+1}/{len(image_paths)}): {filename_base}")

        try:
            # --- Step 1: Load image and extract primary object ROIs ---
            img_pil_orig = Image.open(image_path) # Keep original mode first
            
            # Ensure it has an alpha channel or create a dummy one if it's RGB and we need to treat it as opaque
            if img_pil_orig.mode == 'RGB':
                img_pil_orig = img_pil_orig.convert('RGBA') # Add alpha channel, fully opaque
            elif img_pil_orig.mode != 'RGBA':
                img_pil_orig = img_pil_orig.convert('RGBA') # Convert other modes to RGBA

            img_rgba_orig = np.array(img_pil_orig)


            alpha_channel_full = img_rgba_orig[:, :, 3]
            # Create a binary mask from alpha (255 for object, 0 for background)
            _, binary_mask_from_alpha_full_res = cv2.threshold(alpha_channel_full, 127, 255, cv2.THRESH_BINARY)

            if np.sum(binary_mask_from_alpha_full_res) == 0: # Check if any object pixels exist
                print(f"  Skipping {filename_base}: No foreground in alpha channel.")
                continue

            # Find contours to get the main object
            contours, _ = cv2.findContours(binary_mask_from_alpha_full_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"  Skipping {filename_base}: No contours found from alpha mask.")
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(main_contour)

            # Create a precise ROI mask (filled contour) for the main object, cropped to its bounding box.
            # This mask is uint8, 255 for cat, 0 for background within the bounding box.
            precise_cat_roi_mask_orig_res = np.zeros((h_bbox, w_bbox), dtype=np.uint8)
            shifted_contour = main_contour - [x_bbox, y_bbox] # Adjust contour points to ROI's local coordinates
            cv2.drawContours(precise_cat_roi_mask_orig_res, [shifted_contour], -1, color=255, thickness=cv2.FILLED)

            # Crop the color part (RGBA) of the image using the same bounding box
            cat_color_roi_orig_res = img_rgba_orig[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox, :]


            # --- Step 2: Create and save the filled outline mask ---
            outline_mask_img = create_filled_outline_mask(precise_cat_roi_mask_orig_res, TARGET_SIZE)
            if outline_mask_img is not None:
                save_path_outline = os.path.join(OUTPUT_DIR_OUTLINES, f"{filename_base}_outline.npy")
                np.save(save_path_outline, outline_mask_img)
                # Optional preview for outline:
                cv2.imwrite(os.path.join(OUTPUT_DIR_OUTLINES, f"{filename_base}_outline_preview.png"), (outline_mask_img * 255).astype(np.uint8))
            else:
                print(f"  Failed to create outline mask for {filename_base}. Skipping this image.")
                continue

            # --- Step 3: Create and save the Grayscale Cat image ---
            grayscale_cat_img = create_grayscale_cat_image(cat_color_roi_orig_res, precise_cat_roi_mask_orig_res, TARGET_SIZE)
            if grayscale_cat_img is not None:
                save_path_grayscale = os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{filename_base}_grayscale.npy") # New name
                np.save(save_path_grayscale, grayscale_cat_img)
                # Optional preview for grayscale cat:
                cv2.imwrite(os.path.join(OUTPUT_DIR_GRAYSCALECATS, f"{filename_base}_grayscale_preview.png"), (grayscale_cat_img * 255).astype(np.uint8))
            else:
                print(f"  Failed to create grayscale cat image for {filename_base}. Skipping this image.")
                # Clean up already saved outline if grayscale fails, to maintain pairs
                if os.path.exists(save_path_outline): os.remove(save_path_outline)
                continue
            
            processed_count += 1

        except Exception as e:
            print(f"  Unhandled error processing {filename_base}: {e}")
            continue

    print(f"\nFinished processing. Successfully created paired files for {processed_count} images.")
    print(f"Outline masks saved in: {OUTPUT_DIR_OUTLINES}")
    print(f"Grayscale cat images saved in: {OUTPUT_DIR_GRAYSCALECATS}")

if __name__ == "__main__":
    main()