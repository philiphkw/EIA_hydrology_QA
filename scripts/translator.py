from pathlib import Path
from deep_translator import GoogleTranslator
from tqdm.auto import tqdm
import json
import threading

_position_lock = threading.Lock()
_positions = set()

def get_position():
    with _position_lock:
        pos = next(i for i in range(1, 20) if i not in _positions)
        _positions.add(pos)
        return pos

def release_position(pos):
    with _position_lock:
        _positions.discard(pos)

def translate_text(text):
    chunk_size = 4500
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def translate_files(file_list, txt_dir, progress_dir, out_dir, source_lang, target_lang):

    out_dir.mkdir(exist_ok=True)
    progress_dir.mkdir(exist_ok=True)

    outer = tqdm(total=len(file_list), desc="Total Files", position=0)

    for txt_path in file_list:
        relative = txt_path.relative_to(txt_dir)
        out_file = (out_dir / relative).with_suffix(".txt")
        progress_file = (progress_dir / relative).with_suffix(".json")

        out_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            outer.update(1)
            continue

        try:
            text = txt_path.read_text(encoding="utf-8")
            chunks = translate_text(text)

            if progress_file.exists():
                with open(progress_file) as f:
                    progress = json.load(f)
                translated_chunks = progress["translated_chunks"]
                start_chunk = progress["next_chunk"]
            else:
                translated_chunks = []
                start_chunk = 0

            translator = GoogleTranslator(source=source_lang, target=target_lang)
            pos = get_position()
            BATCH_SIZE = 10

            try:
                with tqdm(range(start_chunk, len(chunks), BATCH_SIZE), desc=relative.name[:40],
                        position=pos, leave=False, total=len(range(start_chunk, len(chunks), BATCH_SIZE))) as pbar:
                    for batch_start in pbar:
                        batch = chunks[batch_start:batch_start + BATCH_SIZE]
                        translated_batch = translator.translate_batch(batch)
                        translated_chunks.extend(translated_batch)
                        with open(progress_file, "w") as f:
                            json.dump({"translated_chunks": translated_chunks, "next_chunk": batch_start + len(batch)}, f)
            finally:
                release_position(pos)

            out_file.write_text("\n".join(translated_chunks), encoding="utf-8")
            progress_file.unlink()
            # tqdm.write(f"✓ Done: {relative}")

        except Exception as e:
            tqdm.write(f"✗ Failed: {relative}: {e}")

        outer.update(1)

    outer.close()
    tqdm.write("All done!")

from deep_translator import GoogleTranslator
from pathlib import Path
from tqdm import tqdm
import time



def translate_folder_names(base_dir, source_lang, target_lang):
    folders = sorted(
        [p for p in base_dir.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True
    )

    names = [f.name for f in folders]

    CHUNK_SIZE = 50
    translated_names = []

    for i in tqdm(range(0, len(names), CHUNK_SIZE), desc="Translating"):
        chunk = names[i:i + CHUNK_SIZE]
        translated_names.extend(GoogleTranslator(source=source_lang, target=target_lang).translate_batch(chunk))
        time.sleep(1)

    for folder, original_name, translated_name in tqdm(
        zip(folders, names, translated_names), total=len(folders), desc="Renaming"
    ):
        try:
            if translated_name is None or translated_name == original_name:
                tqdm.write(f"⏭ Unchanged: {original_name}")
                continue

            for char in r'<>:"/\|?*':
                translated_name = translated_name.replace(char, "")

            new_path = folder.parent / translated_name
            folder.rename(new_path)
            tqdm.write(f"✓ {original_name} → {translated_name}")

        except Exception as e:
            tqdm.write(f"✗ Failed: {original_name}: {e}")

    print("All done!")

    return


def translate_file_names(dir, source_lang, target_lang):
    files = list(dir.rglob("*.txt"))
    stems = [f.stem for f in files]

    def clean_filename(name):
        for char in r'<>:"/\|?*':
            name = name.replace(char, "")
        return name.strip()

    CHUNK_SIZE = 50
    translated_stems = []
    for i in tqdm(range(0, len(stems), CHUNK_SIZE), desc="Translating"):
        chunk = stems[i:i + CHUNK_SIZE]
        results = GoogleTranslator(source=source_lang, target=target_lang).translate_batch(chunk)
        translated_stems.extend([clean_filename(r) for r in results])
        time.sleep(1)

    for file, original_stem, translated_stem in tqdm(
        zip(files, stems, translated_stems), total=len(files), desc="Renaming"
    ):
        try:
            if translated_stem is None or translated_stem == original_stem:
                tqdm.write(f"⏭ Unchanged: {original_stem}")
                continue

            new_path = file.parent / (translated_stem + file.suffix)
            file.rename(new_path)
            tqdm.write(f"✓ {original_stem} → {translated_stem}")

        except Exception as e:
            tqdm.write(f"✗ Failed: {original_stem}: {e}")

    print("\nAll done!")

    return