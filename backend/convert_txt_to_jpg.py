import os
import re

BARGAD_DIR = "dataset/bargad"

converted = 0
failed = 0

for person in os.listdir(BARGAD_DIR):
    person_dir = os.path.join(BARGAD_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for file in os.listdir(person_dir):
        if file.endswith(".txt"):
            txt_path = os.path.join(person_dir, file)
            jpg_path = os.path.join(person_dir, file.replace(".txt", ".jpg"))
            try:
                with open(txt_path, "rb") as f:
                    raw = f.read()

                raw_str = raw.decode("utf-8", errors="ignore").strip()
                raw_str = raw_str.replace("\n", "").replace("\r", "").replace(" ", "")
                raw_str = raw_str.lstrip("\ufeff")

                # Strip 0x prefix
                if raw_str.lower().startswith("0x"):
                    raw_str = raw_str[2:]

                # Remove ALL non-hex characters
                raw_str = re.sub(r'[^0-9a-fA-F]', '', raw_str)

                # Ensure even length
                if len(raw_str) % 2 != 0:
                    raw_str = raw_str[:-1]

                img_bytes = bytes.fromhex(raw_str)

                with open(jpg_path, "wb") as f:
                    f.write(img_bytes)

                os.remove(txt_path)
                converted += 1
                print(f"✅ Converted: {file}")

            except Exception as e:
                failed += 1
                print(f"❌ Failed: {file} → {e}")

print(f"\n🎉 Done! Converted {converted} files. Failed: {failed}")
