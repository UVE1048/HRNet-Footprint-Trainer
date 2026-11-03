import os
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
import cv2
from tqdm import tqdm
import ast
from pathlib import Path
import shutil
from multiprocessing import Pool, cpu_count
import time
import glob

# ================= 配置参数 =================
CROP_SIZE = 1024
STRIDE = 512                 # 50% overlap
OUTPUT_DIR = r"../dataset"
TRAIN_RATIO = 0.8
PIXEL_VALUE_BUILDING = 1
MIN_FG_RATIO = 0.01         # 建筑像素 ≥ 1%
MAX_FG_RATIO = 0.98         # 过滤纯建筑 (fg > 98%)
NUM_PROCESSES = min(cpu_count(), 32)

# ==========================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data_config(config_path):
    if not os.path.exists(config_path):
        tqdm.write(f"Error: Config file not found: {config_path}")
        return []
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            if "datasets =" in content:
                content = content.split("datasets =")[1].strip()
            return ast.literal_eval(content)
        except Exception as e:
            tqdm.write(f"Error parsing DATA.TXT: {e}")
            return []

def fix_path(path):
    if os.path.exists(path):
        return path
    new_path = path.replace("Footprint_Trainer", "HRNet_Footprint_Trainer")
    if os.path.exists(new_path):
        return new_path
    return path

def save_patch(args):
    img_crop, mask_crop, output_root, dataset_name, x, y = args
    filename = f"{dataset_name}_{x}_{y}.png"

    img_out_path = os.path.join(output_root, "images", filename)
    mask_out_path = os.path.join(output_root, "masks", filename)
    mask_vis_path = os.path.join(output_root, "masks_vis", filename)

    mask_crop = mask_crop.astype(np.uint8)

    cv2.imwrite(img_out_path, cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_out_path, mask_crop)
    cv2.imwrite(mask_vis_path, mask_crop * 255)

    return img_out_path, mask_out_path

def process_single_pair(tif_path, shp_path, output_root, dataset_name, existing_files):
    if not os.path.exists(tif_path) or not os.path.exists(shp_path):
        tqdm.write(f"[{dataset_name}] Skipping: File not found.")
        return []

    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        meta = src.meta
        nodata = src.nodata

    ys = list(range(0, height - CROP_SIZE + 1, STRIDE))
    xs = list(range(0, width  - CROP_SIZE + 1, STRIDE))
    if ys[-1] != height - CROP_SIZE:
        ys.append(height - CROP_SIZE)
    if xs[-1] != width - CROP_SIZE:
        xs.append(width - CROP_SIZE)

    # Filter tasks based on existing files to avoid reprocessing
    potential_tasks = []
    skipped_existing = 0
    for y in ys:
        for x in xs:
            filename = f"{dataset_name}_{x}_{y}.png"
            if filename in existing_files:
                skipped_existing += 1
            else:
                potential_tasks.append((x, y))

    if not potential_tasks:
        if skipped_existing > 0:
            tqdm.write(f"[{dataset_name}] All {skipped_existing} patches already exist. Skipping.")
        return []

    tqdm.write(f"[{dataset_name}] Rasterizing mask...")
    gdf = gpd.read_file(shp_path)
    if gdf.crs != meta['crs']:
        gdf = gdf.to_crs(meta['crs'])

    mask = rasterize(
        ((geom, PIXEL_VALUE_BUILDING) for geom in gdf.geometry),
        out_shape=(height, width),
        fill=0,
        transform=transform,
        dtype=np.uint8
    )

    ensure_dir(os.path.join(output_root, "images"))
    ensure_dir(os.path.join(output_root, "masks"))
    ensure_dir(os.path.join(output_root, "masks_vis"))

    tqdm.write(f"[{dataset_name}] Checking content for {len(potential_tasks)} patches...")
    valid_tasks = []
    
    for x, y in potential_tasks:
        mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
        fg_ratio = (mask_crop > 0).mean()
        
        if fg_ratio < MIN_FG_RATIO:
            continue
        if fg_ratio > MAX_FG_RATIO:
            continue
            
        valid_tasks.append((x, y))

    total_tasks = len(valid_tasks)
    if skipped_existing > 0:
        tqdm.write(f"[{dataset_name}] Skipped {skipped_existing} existing files. Found {total_tasks} new valid patches.")
    else:
        tqdm.write(f"[{dataset_name}] Found {total_tasks} valid patches.")

    if total_tasks == 0:
        return []

    def task_generator(nodata_val):
        with rasterio.open(tif_path) as src:
            for x, y in valid_tasks:
                window = Window(x, y, CROP_SIZE, CROP_SIZE)
                img_window = src.read(window=window)

                if nodata_val is not None:
                    img_window = np.where(img_window == nodata_val, 0, img_window)

                if img_window.shape[0] >= 3:
                    img_crop = img_window[:3].transpose(1, 2, 0)
                else:
                    gray = img_window[0]
                    img_crop = np.stack([gray, gray, gray], axis=-1)

                # Handle implicit nodata: 0,0,0 or 255,255,255 -> 0,0,0
                # Primarily for 8-bit RGB where 255 is white background
                if img_crop.dtype == np.uint8 or np.max(img_crop) <= 255:
                    # Check for white (255, 255, 255) and set to 0
                    is_white = np.all(img_crop == 255, axis=-1)
                    if np.any(is_white):
                        img_crop[is_white] = 0
                    # (0,0,0) is already 0, so no action needed

                if img_crop.dtype != np.uint8:
                    max_val = np.max(img_crop)
                    if max_val <= 255:
                        img_crop = img_crop.astype(np.uint8)
                    else:
                        p_min = np.percentile(img_crop, 2)
                        p_max = np.percentile(img_crop, 98)
                        if p_max > p_min:
                            img_crop = np.clip((img_crop - p_min) / (p_max - p_min) * 255, 0, 255).astype(np.uint8)
                        else:
                            img_crop = np.clip(img_crop, 0, 255).astype(np.uint8)

                mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
                
                yield (img_crop, mask_crop, output_root, dataset_name, x, y)

    results = []
    with Pool(processes=NUM_PROCESSES) as pool:
        for res in tqdm(pool.imap_unordered(save_patch, task_generator(nodata)),
                        total=total_tasks,
                        desc=f"Saving {dataset_name}", leave=False):
            results.append(res)

    return results

def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    import argparse
    ap = argparse.ArgumentParser(
        description="Slice large orthophotos + building vectors into training patches."
    )
    ap.add_argument(
        "--config",
        default=os.path.join(base_dir, "DATA.TXT"),
        help="dataset list file (default: prepare/DATA.TXT, copy it from DATA.example.TXT)",
    )
    cli_args, _ = ap.parse_known_args()
    data_config_path = cli_args.config

    datasets = load_data_config(data_config_path)
    if not datasets:
        tqdm.write("No datasets found.")
        return

    root_out = os.path.join(base_dir, OUTPUT_DIR)
    ensure_dir(root_out)

    # Initialize train/val directories
    train_dir = os.path.join(root_out, "train")
    val_dir = os.path.join(root_out, "val")
    ensure_dir(os.path.join(train_dir, "images"))
    ensure_dir(os.path.join(train_dir, "masks"))
    ensure_dir(os.path.join(val_dir, "images"))
    ensure_dir(os.path.join(val_dir, "masks"))

    # Scan for existing files
    tqdm.write("Scanning existing dataset files...")
    existing_files = set()
    for d in [train_dir, val_dir]:
        img_dir = os.path.join(d, "images")
        if os.path.exists(img_dir):
            existing_files.update(os.listdir(img_dir))
    tqdm.write(f"Found {len(existing_files)} existing patches. These will be skipped.")

    temp_dir = os.path.join(root_out, "temp")
    ensure_dir(temp_dir)

    all_new_pairs = []

    for tif_path, shp_path in tqdm(datasets, desc="Total Progress"):
        tif_path = fix_path(tif_path)
        shp_path = fix_path(shp_path)

        dataset_name = os.path.splitext(os.path.basename(tif_path))[0]
        # Pass existing_files to skip duplicates
        pair_results = process_single_pair(tif_path, shp_path, temp_dir, dataset_name, existing_files)
        all_new_pairs.extend(pair_results)

    # Process only new pairs
    if all_new_pairs:
        tqdm.write(f"Processing {len(all_new_pairs)} new patches...")
        random.seed(42)
        random.shuffle(all_new_pairs)
        split_idx = int(len(all_new_pairs) * TRAIN_RATIO)

        train_pairs = all_new_pairs[:split_idx]
        val_pairs = all_new_pairs[split_idx:]

        def move_files(pairs, target_dir):
            for img_p, mask_p in tqdm(pairs, desc=f"Moving to {os.path.basename(target_dir)}"):
                fname = os.path.basename(img_p)
                
                dst_img = os.path.join(target_dir, "images", fname)
                dst_mask = os.path.join(target_dir, "masks", fname)
                
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                shutil.move(img_p, dst_img)
                
                if os.path.exists(dst_mask):
                    os.remove(dst_mask)
                shutil.move(mask_p, dst_mask)

                vis_p = mask_p.replace("masks", "masks_vis")
                if os.path.exists(vis_p):
                    vis_dir = os.path.join(target_dir, "masks_vis")
                    ensure_dir(vis_dir)
                    dst_vis = os.path.join(vis_dir, fname)
                    if os.path.exists(dst_vis):
                        os.remove(dst_vis)
                    shutil.move(vis_p, dst_vis)

        move_files(train_pairs, train_dir)
        move_files(val_pairs, val_dir)
    else:
        tqdm.write("No new patches to add.")

    shutil.rmtree(temp_dir, ignore_errors=True)

    # Regenerate lists using ALL files (old + new)
    tqdm.write("Regenerating dataset lists...")
    def generate_list(target_dir, list_name):
        list_file = os.path.join(root_out, list_name)
        # Use glob to find all images
        imgs = sorted(glob.glob(os.path.join(target_dir, "images", "*.png")))
        
        with open(list_file, 'w') as f:
            for img in imgs:
                rel_img = os.path.relpath(img, root_out).replace("\\", "/")
                rel_mask = rel_img.replace("images", "masks")
                f.write(f"{rel_img} {rel_mask}\n")
        return len(imgs)

    n_train = generate_list(train_dir, "train_list.txt")
    n_val = generate_list(val_dir, "val_list.txt")

    tqdm.write(f"Done! Total time: {time.time() - start_time:.2f}s")
    tqdm.write(f"Total Train: {n_train}, Total Val: {n_val}")

if __name__ == "__main__":
    main()
