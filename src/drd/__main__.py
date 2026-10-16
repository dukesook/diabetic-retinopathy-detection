from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Returns the root directory of the repository."""
    # Navigates up from src/drd/__main__.py to project root
    return Path(__file__).resolve().parent.parent.parent


def load_dataset_metadata():
    root = get_project_root()
    data_dir = root / "data"
    csv_path = data_dir / "trainLabels.csv"
    images_dir = data_dir / "sample"

    # 1. Verify files and directories exist
    if not csv_path.exists():
        raise FileNotFoundError(f"Label file not found at: {csv_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    print(f"Loading labels from: {csv_path.relative_to(root)}")

    # 2. Read CSV into Pandas DataFrame
    df = pd.read_csv(csv_path)

    print("\n--- DataFrame Summary ---")
    print(f"Total entries in CSV: {len(df)}")
    print("First 5 rows:")
    print(df.head())

    # 3. Add full file path column for CNN data loaders
    # Assumes CSV has columns like 'image' (e.g., '10_left') and 'level' (label)
    df["image_path"] = df["image"].apply(
        lambda img_name: images_dir / f"{img_name}.jpeg"
    )

    # 4. Check image existence in sample folder
    df["exists"] = df["image_path"].apply(lambda p: p.exists())
    available_samples = df[df["exists"]].copy()

    print(f"\nImages found in '{images_dir.name}/': {len(available_samples)}")

    if available_samples.empty:
        print(
            "Warning: None of the CSV entries match images in the sample folder."
        )
    else:
        print("\nMatched Sample Images preview:")
        print(
            available_samples[["image", "level", "image_path"]].head()
        )

    return available_samples


def main():
    print("===== Diabetic Retinopathy Detection =====\n")
    df = load_dataset_metadata()


if __name__ == "__main__":
    main()