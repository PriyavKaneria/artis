# src/prepare_data.py
import cv2
import numpy as np
from PIL import Image
import os
import glob

TARGET_SIZE = (128, 128) # Or (64, 64) if you revert
OUTPUT_DIR_BASE = "dataset/processed/pixel_art_cat"
OUTPUT_DIR_OUTLINES = os.path.join(OUTPUT_DIR_BASE, "outline_masks")
OUTPUT_DIR_EDGEDETAILS = os.path.join(OUTPUT_DIR_BASE, "edge_details")
INPUT_DIR_PNGS = "dataset/cat"

# --- Parameters for Edge Enhancement ---
# 1. Gaussian Blur (applied before Canny)
#    Kernel size must be positive and odd. (5,5) is a common starting point.
#    Increase for more smoothing, decrease for less. Set to (0,0) or None to disable.
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5) 
GAUSSIAN_BLUR_SIGMAX = 0 # Auto-calculates sigma based on kernel size if 0

# 2. Canny Edge Detection Parameters
#    Lower threshold2 makes Canny more sensitive, detecting more (potentially weaker) edges.
#    Threshold1 is typically 0.4 or 0.5 of threshold2.
#    YOU WILL LIKELY NEED TO TUNE THESE BASED ON YOUR IMAGES AND BLUR SETTINGS.
CANNY_THRESHOLD1 = 30  # Lowered from 50
CANNY_THRESHOLD2 = 100 # Lowered from 150

# 3. Dilation (applied after Canny)
#    This makes detected edges thicker and can connect small gaps.
DILATION_KERNEL_SIZE = (2, 2) # A small kernel like 2x2 or 3x3
DILATION_ITERATIONS = 1       # Number of times dilation is applied. Increase for thicker lines.
                              # Set to 0 to disable dilation.
# --- End of Edge Enhancement Parameters ---


SCALING_PADDING = 4 # Pixels of padding when scaling to fit TARGET_SIZE

def create_filled_outline_mask(precise_cat_roi_mask_orig_res, target_size):
    """
    Creates the filled B&W outline mask. Cat is 0 (black), background is 1 (white).
    precise_cat_roi_mask_orig_res: The cat's shape (255 for cat, 0 for bg) cropped to its bounding box from original resolution.
    """
    if precise_cat_roi_mask_orig_res is None or precise_cat_roi_mask_orig_res.size == 0:
        return None

    outline_image_final = np.ones(target_size, dtype=np.float32)
    
    roi_h, roi_w = precise_cat_roi_mask_orig_res.shape
    if roi_w == 0 or roi_h == 0: return None

    scale = min((target_size[0] - SCALING_PADDING) / roi_w, (target_size[1] - SCALING_PADDING) / roi_h)
    if scale <= 0: return None
        
    new_w, new_h = max(1, int(roi_w * scale)), max(1, int(roi_h * scale))
    resized_cat_roi_mask = cv2.resize(precise_cat_roi_mask_orig_res, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    start_x = (target_size[0] - new_w) // 2
    start_y = (target_size[1] - new_h) // 2
    outline_image_final[start_y:start_y+new_h, start_x:start_x+new_w][resized_cat_roi_mask == 255] = 0.0
    return outline_image_final


def create_bnw_edge_detail_image(cat_color_roi_orig_res, precise_cat_roi_mask_orig_res, target_size):
    """
    Creates B&W edge detail image. Edges are 0 (black), background is 1 (white).
    cat_color_roi_orig_res: The RGB part of the cat, cropped to its bounding box.
    precise_cat_roi_mask_orig_res: The cat's precise shape (0 or 255) within that bounding box, used for masking.
    """
    if cat_color_roi_orig_res is None or precise_cat_roi_mask_orig_res is None:
        return None
    
    gray_cat_color_roi = cv2.cvtColor(cat_color_roi_orig_res, cv2.COLOR_RGB2GRAY)
    masked_gray_for_canny_input = cv2.bitwise_and(gray_cat_color_roi, gray_cat_color_roi, mask=precise_cat_roi_mask_orig_res)

    # --- Apply Gaussian Blur ---
    blurred_img = masked_gray_for_canny_input
    if GAUSSIAN_BLUR_KERNEL_SIZE and GAUSSIAN_BLUR_KERNEL_SIZE[0] > 0 and GAUSSIAN_BLUR_KERNEL_SIZE[1] > 0:
        blurred_img = cv2.GaussianBlur(masked_gray_for_canny_input, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMAX)

    # --- Canny Edge Detection ---
    edges_canny = cv2.Canny(blurred_img, CANNY_THRESHOLD1, CANNY_THRESHOLD2) # Edges are 255 (white) on black bg

    # --- Apply Dilation ---
    dilated_edges = edges_canny
    if DILATION_ITERATIONS > 0 and DILATION_KERNEL_SIZE and DILATION_KERNEL_SIZE[0] > 0 and DILATION_KERNEL_SIZE[1] > 0:
        dilation_kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
        dilated_edges = cv2.dilate(edges_canny, dilation_kernel, iterations=DILATION_ITERATIONS)
    
    # --- Prepare final canvas and place edges ---
    edge_detail_image_final = np.ones(target_size, dtype=np.float32) # White background (1.0)

    roi_h, roi_w = precise_cat_roi_mask_orig_res.shape # Scale based on original mask shape for consistency
    if roi_w == 0 or roi_h == 0: return None

    scale = min((target_size[0] - SCALING_PADDING) / roi_w, (target_size[1] - SCALING_PADDING) / roi_h)
    if scale <= 0: return None
        
    new_w, new_h = max(1, int(roi_w * scale)), max(1, int(roi_h * scale))

    # Resize the (potentially dilated) Canny edges
    resized_edges = cv2.resize(dilated_edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    start_x = (target_size[0] - new_w) // 2
    start_y = (target_size[1] - new_h) // 2

    # Where resized_edges is 255 (edge), make it 0.0 (black) in the final image
    edge_detail_image_final[start_y:start_y+new_h, start_x:start_x+new_w][resized_edges == 255] = 0.0
    
    return edge_detail_image_final


def main():
    os.makedirs(OUTPUT_DIR_OUTLINES, exist_ok=True)
    os.makedirs(OUTPUT_DIR_EDGEDETAILS, exist_ok=True)

    image_paths = glob.glob(os.path.join(INPUT_DIR_PNGS, "*.png"))
    if not image_paths:
        print(f"No PNG files found in {INPUT_DIR_PNGS}. Please check the path.")
        return

    print(f"Found {len(image_paths)} PNG files. Starting processing to {TARGET_SIZE}...")
    print(f"Using Blur Kernel: {GAUSSIAN_BLUR_KERNEL_SIZE}, Canny Thresh: ({CANNY_THRESHOLD1},{CANNY_THRESHOLD2}), Dilation Kernel: {DILATION_KERNEL_SIZE} Iter: {DILATION_ITERATIONS}")

    processed_count = 0
    for i, image_path in enumerate(image_paths):
        filename_base = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing ({i+1}/{len(image_paths)}): {filename_base}")

        try:
            img_pil_orig = Image.open(image_path).convert("RGBA")
            img_rgba_orig = np.array(img_pil_orig)

            if img_rgba_orig.shape[2] != 4:
                print(f"  Skipping {filename_base}: Not an RGBA image.")
                continue

            alpha_channel_full = img_rgba_orig[:, :, 3]
            _, binary_mask_from_alpha_full_res = cv2.threshold(alpha_channel_full, 127, 255, cv2.THRESH_BINARY)

            if np.sum(binary_mask_from_alpha_full_res) == 0:
                print(f"  Skipping {filename_base}: No foreground in alpha channel.")
                continue

            contours, _ = cv2.findContours(binary_mask_from_alpha_full_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print(f"  Skipping {filename_base}: No contours found from alpha mask.")
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(main_contour)

            precise_cat_roi_mask_orig_res = np.zeros((h_bbox, w_bbox), dtype=np.uint8)
            shifted_contour = main_contour - [x_bbox, y_bbox]
            cv2.drawContours(precise_cat_roi_mask_orig_res, [shifted_contour], -1, color=255, thickness=cv2.FILLED)

            cat_color_roi_orig_res = img_rgba_orig[y_bbox:y_bbox+h_bbox, x_bbox:x_bbox+w_bbox, :3]

            outline_mask_img = create_filled_outline_mask(precise_cat_roi_mask_orig_res, TARGET_SIZE)
            if outline_mask_img is not None:
                save_path_outline = os.path.join(OUTPUT_DIR_OUTLINES, f"{filename_base}_outline.npy")
                np.save(save_path_outline, outline_mask_img)
                # Optional preview for outline:
                cv2.imwrite(os.path.join(OUTPUT_DIR_OUTLINES, f"{filename_base}_outline_preview.png"), (outline_mask_img * 255).astype(np.uint8))
            else:
                print(f"  Failed to create outline mask for {filename_base}. Skipping this image.")
                continue

            edge_detail_img = create_bnw_edge_detail_image(cat_color_roi_orig_res, precise_cat_roi_mask_orig_res, TARGET_SIZE)
            if edge_detail_img is not None:
                save_path_edge = os.path.join(OUTPUT_DIR_EDGEDETAILS, f"{filename_base}_edgedetail.npy")
                np.save(save_path_edge, edge_detail_img)
                # Optional preview for edge detail:
                cv2.imwrite(os.path.join(OUTPUT_DIR_EDGEDETAILS, f"{filename_base}_edgedetail_preview.png"), (edge_detail_img * 255).astype(np.uint8))
            else:
                print(f"  Failed to create edge detail image for {filename_base}. Skipping this image.")
                if os.path.exists(save_path_outline): os.remove(save_path_outline)
                continue
            
            processed_count += 1

        except Exception as e:
            print(f"  Unhandled error processing {filename_base}: {e}")
            continue

    print(f"\nFinished processing. Successfully created paired files for {processed_count} images.")
    print(f"Outline masks saved in: {OUTPUT_DIR_OUTLINES}")
    print(f"Edge detail images saved in: {OUTPUT_DIR_EDGEDETAILS}")

if __name__ == "__main__":
    main()