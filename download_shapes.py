import os
import requests
import zipfile

SHAPES_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/wzr2yv7r53-1.zip"
DATA_DIR   = "/oscar/home/apate206/data/students/apate206/Computer-Vision-Final-Project/shapes"
ZIP_PATH   = os.path.join(DATA_DIR, "shapes.zip")

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading Mendeley 2D Geometric Shapes dataset...")
with requests.get(SHAPES_URL, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb  = downloaded / 1e6
                print(f"\r  {mb:.1f} MB / {total/1e6:.1f} MB  ({pct}%)", end="")
print("\nDownload complete.")

print("Extracting dataset...")
with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    members = [m for m in zf.namelist() if m.endswith(".png")]
    print(f"  {len(members)} PNG files found in archive")
    for i, member in enumerate(members):
        # Extract flat into DATA_DIR (strip any subfolder structure)
        fname = os.path.basename(member)
        if not fname:
            continue
        data = zf.read(member)
        with open(os.path.join(DATA_DIR, fname), "wb") as out:
            out.write(data)
        if i % 1000 == 0:
            print(f"\r  Extracted {i}/{len(members)}", end="")
print(f"\r  Extracted {len(members)}/{len(members)} files")
print("Extraction complete.")

# Remove zip to save space
os.remove(ZIP_PATH)
print("Cleanup complete.")

final = len([f for f in os.listdir(DATA_DIR) if f.endswith(".png")])
print(f"\nDone! {final} PNG files in {DATA_DIR}")
print(f'\nIn your notebook set:\n  SHAPES_DIR = "{DATA_DIR}"')
