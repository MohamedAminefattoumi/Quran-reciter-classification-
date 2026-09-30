"""Exploratory data analysis for the Quran recitations audio dataset."""

import re
import shutil
from collections import Counter
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf

DATASET_ID = "mohammedalrajeh/quran-recitations-for-audio-classification"
PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_DIR / "data"


# ----------------------------------------------------------------------
# 1. Dataset access
# ----------------------------------------------------------------------
def count_files(folder):
    """Count all files inside a folder (recursively)."""
    return sum(1 for p in folder.rglob("*") if p.is_file())


def get_local_dataset():
    """Return the dataset folder inside the project.

    On the first run, the dataset is downloaded with kagglehub, copied into
    the project, verified, and then the kagglehub cache copy is deleted.
    """
    if LOCAL_DATA_DIR.exists():
        print("Dataset already in project:", LOCAL_DATA_DIR)
        return LOCAL_DATA_DIR

    cache_path = Path(kagglehub.dataset_download(DATASET_ID))
    print("Dataset found in cache:", cache_path)

    # Copy into a temporary folder first, so an interrupted copy never
    # leaves a half-filled "data" folder behind
    tmp_dir = PROJECT_DIR / "data_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("Copying dataset into the project (this can take a while)...")
    shutil.copytree(cache_path, tmp_dir)

    # Verify the copy before touching the cache
    n_source, n_copy = count_files(cache_path), count_files(tmp_dir)
    if n_source != n_copy:
        raise RuntimeError(
            f"Copy incomplete: {n_copy}/{n_source} files. Cache was NOT deleted."
        )

    tmp_dir.rename(LOCAL_DATA_DIR)
    print(f"Copy verified: {n_copy} files in {LOCAL_DATA_DIR}")

    # Delete the whole cache folder of this dataset (versions/1 and its marker files)
    cache_dataset_dir = cache_path.parent.parent
    try:
        shutil.rmtree(cache_dataset_dir)
        print("Cache copy deleted:", cache_dataset_dir)
    except OSError as e:
        print("Could not delete the cache copy:", e)

    return LOCAL_DATA_DIR


def get_dataset_paths():
    """Return the dataset root, the audio folder and the CSV path."""
    root = get_local_dataset()
    audio_dir = root / "Dataset"  # CSV paths are relative to this folder
    csv_path = root / "files_paths.csv"
    return root, audio_dir, csv_path


def inspect_folder(root):
    """Print the top-level structure and the file extensions of the dataset."""
    print("\n=== Folder structure ===")
    for item in sorted(root.iterdir()):
        print("DIR " if item.is_dir() else "FILE", item.name)

    extensions = Counter(p.suffix.lower() for p in root.rglob("*") if p.is_file())
    print("File extensions:", dict(extensions))


# ----------------------------------------------------------------------
# 2. Metadata loading and integrity checks
# ----------------------------------------------------------------------
def load_metadata(csv_path, audio_dir):
    """Load the CSV and add absolute paths and a noiseRed flag."""
    df = pd.read_csv(csv_path)
    print("\n=== CSV ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())

    df["abs_path"] = df["FilePath"].apply(lambda p: audio_dir / p)
    df["is_noiseRed"] = df["FilePath"].str.contains("noiseRed")
    return df


def check_integrity(df, audio_dir):
    """Check the CSV against the wav files on disk and add any missing rows."""
    print("\n=== Integrity checks ===")

    # CSV entries that point to a file that does not exist
    missing = df[~df["abs_path"].apply(lambda p: p.exists())]
    print("Files listed in CSV but missing on disk:", len(missing))

    # wav files on disk that are not listed in the CSV
    all_wavs = {p.resolve() for p in audio_dir.rglob("*.wav")}
    csv_wavs = {p.resolve() for p in df["abs_path"]}
    unlisted = sorted(all_wavs - csv_wavs)
    print("wav files not in CSV:", [p.name for p in unlisted])

    # Add the unlisted files (class = name of the parent folder)
    if unlisted:
        extra = pd.DataFrame(
            {
                "FilePath": ["./" + p.relative_to(audio_dir.resolve()).as_posix() for p in unlisted],
                "Class": [p.parent.name for p in unlisted],
                "abs_path": unlisted,
                "is_noiseRed": ["noiseRed" in p.name for p in unlisted],
            }
        )
        df = pd.concat([df, extra], ignore_index=True)
        print("Added unlisted files. New total:", len(df))

    return df


# ----------------------------------------------------------------------
# 3. Feature extraction from file names and audio headers
# ----------------------------------------------------------------------
def add_file_id(df):
    """Extract the trailing number of each file name (e.g. lohaidan_171.wav -> 171)."""
    df["file_id"] = df["abs_path"].apply(
        lambda p: int(re.search(r"(\d+)\.wav$", p.name).group(1))
    )
    return df


def add_audio_properties(df):
    """Read duration, sample rate and channel count from each wav header."""
    props = [sf.info(p) for p in df["abs_path"]]
    df["duration"] = [i.duration for i in props]
    df["sr"] = [i.samplerate for i in props]
    df["channels"] = [i.channels for i in props]
    df["duration_r"] = df["duration"].round(1)
    return df


# ----------------------------------------------------------------------
# 4. Reports
# ----------------------------------------------------------------------
def report_class_distribution(df):
    print("\n=== Class distribution ===")
    print(f"Data shape: {df.shape}")
    print("Classes:", list(df["Class"].unique()))
    print("Number of classes:", df["Class"].nunique())
    print(df["Class"].value_counts())
    print("\nFiles per class and type (noiseRed vs original):")
    print(pd.crosstab(df["Class"], df["is_noiseRed"]))


def report_audio_properties(df):
    print("\n=== Audio properties ===")
    print(
        df.groupby("is_noiseRed")[["duration", "sr", "channels"]].agg(
            ["min", "max", "nunique"]
        )
    )
    print("\nDurations (rounded):")
    print(df.groupby(["is_noiseRed", "duration_r"]).size())
    print("\nChannels:")
    print(df.groupby(["is_noiseRed", "channels"]).size())


def plot_class_distribution(df, out_path="class_distribution.png"):
    """Save a stacked bar chart of files per class (original vs noiseRed)."""
    counts = pd.crosstab(df["Class"], df["is_noiseRed"])
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    counts.plot(kind="barh", stacked=True, figsize=(9, 6))
    plt.xlabel("Number of files")
    plt.title("Files per class")
    plt.legend(["original", "noiseRed"])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved plot:", out_path)


def plot_class_pie(df, out_path="class_pie.png", show=True):
    """Save (and optionally show) a pie chart of the share of files per class."""
    counts = df["Class"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
    plt.title("Class Distribution")
    plt.savefig(out_path)
    print("Saved plot:", out_path)
    if show:
        plt.show()  # blocks until the window is closed
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    root, audio_dir, csv_path = get_dataset_paths()
    inspect_folder(root)

    df = load_metadata(csv_path, audio_dir)
    df = check_integrity(df, audio_dir)
    df = add_file_id(df)
    df = add_audio_properties(df)

    report_class_distribution(df)
    report_audio_properties(df)
    plot_class_distribution(df)
    plot_class_pie(df)  # last one: plt.show() blocks until the window is closed

    return df


if __name__ == "__main__":
    df = main()