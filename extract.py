import os
import json
import pandas as pd

from app import process_video


VIDEO_EXTENSIONS = (".mp4", ".MOV", ".mov", ".avi", ".mkv")


def batch_process(video_root, output_csv="dataset.csv"):
    rows = []

    clip_number = 0

    for label in os.listdir(video_root):
        label_path = os.path.join(video_root, label)

        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if not filename.lower().endswith(VIDEO_EXTENSIONS):
                continue

            video_path = os.path.join(label_path, filename)

            print(f"Processing {video_path}...")

            try:
                result = process_video(video_path, filename)

                row = {
                    "video": filename,
                    "path": video_path,
                    "label": label,
                    **result
                }

                rows.append(row)
                clip_number += 1

            except Exception as e:
                print(f"FAILED: {video_path}")
                print(e)

    # JSON preserves nested landmarks better
    # with open(output_json, "w") as f:
    #     json.dump(rows, f, indent=2)

    # CSV is easier for quick ML/table viewing, but nested lists get ugly
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(rows)} clips")
    print(f"CSV: {output_csv}")
    # print(f"JSON: {output_json}")


if __name__ == "__main__":
    batch_process("dataset_videos")