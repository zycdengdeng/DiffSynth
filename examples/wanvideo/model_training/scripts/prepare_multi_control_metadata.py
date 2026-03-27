"""
Generate metadata.csv for multi-control training from the dataset structure:
  BlurProjection/  (sparse color projection)
  DepthSparse/     (sparse depth)
  HDMapBbox/       (hdmap + 3d bbox)

Each row maps: GT video + 3 control videos + caption text
"""
import os
import json
import csv
import argparse
from pathlib import Path


CAMERAS = [
    "ftheta_camera_cross_left_120fov",
    "ftheta_camera_cross_right_120fov",
    "ftheta_camera_front_tele_30fov",
    "ftheta_camera_front_wide_120fov",
    "ftheta_camera_rear_left_70fov",
    "ftheta_camera_rear_right_70fov",
    "ftheta_camera_rear_tele_30fov",
]


def load_caption(caption_path):
    with open(caption_path, "r") as f:
        data = json.load(f)
    return data["caption"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root path, e.g. /mnt/zihanw/proj_utils_pro/transfer_video_maker/output_full_data")
    parser.add_argument("--output_csv", type=str, default="metadata.csv")
    args = parser.parse_args()

    root = Path(args.data_root)

    blur_control_dir = "control_input_blur"
    depth_control_dir = "control_input_depth"
    bbox_control_dir = "control_input_hdmap_bbox"

    rows = []
    missing = []

    for camera in CAMERAS:
        # Use BlurProjection/videos as the canonical GT source
        gt_dir = root / "BlurProjection" / "videos" / camera
        if not gt_dir.exists():
            print(f"Warning: GT dir not found: {gt_dir}")
            continue

        for video_file in sorted(gt_dir.glob("*.mp4")):
            clip_name = video_file.stem  # e.g. "001_seg01"

            # Paths (relative to data_root for portability)
            gt_path = f"BlurProjection/videos/{camera}/{clip_name}.mp4"
            blur_path = f"BlurProjection/{blur_control_dir}/{camera}/{clip_name}.mp4"
            depth_path = f"DepthSparse/{depth_control_dir}/{camera}/{clip_name}.mp4"
            bbox_path = f"HDMapBbox/{bbox_control_dir}/{camera}/{clip_name}.mp4"
            caption_path = root / "BlurProjection" / "captions" / camera / f"{clip_name}.json"

            # Check all files exist
            all_exist = True
            for label, p in [("blur", blur_path), ("depth", depth_path), ("bbox", bbox_path)]:
                if not (root / p).exists():
                    missing.append(f"{label}: {p}")
                    all_exist = False

            if not all_exist:
                continue

            # Load caption
            if caption_path.exists():
                prompt = load_caption(caption_path)
            else:
                prompt = f"a driving scene from {camera.replace('ftheta_camera_', '').replace('_', ' ')} camera"

            rows.append({
                "video": gt_path,
                "control_video_sparse_depth": depth_path,
                "control_video_sparse_color": blur_path,
                "control_video_bbox": bbox_path,
                "prompt": prompt,
            })

    # Write CSV
    output_path = os.path.join(args.data_root, args.output_csv)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "control_video_sparse_depth", "control_video_sparse_color", "control_video_bbox", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nGenerated {len(rows)} samples -> {output_path}")
    if missing:
        print(f"Skipped {len(missing)} samples due to missing files (first 5):")
        for m in missing[:5]:
            print(f"  {m}")


if __name__ == "__main__":
    main()
