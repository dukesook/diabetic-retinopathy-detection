from pathlib import Path
from zipfile import ZipFile

import pandas as pd


def get_data_dir() -> Path:
    """Return the KaggleHub cache directory for this competition."""
    return (
        Path.home()
        / ".cache"
        / "kagglehub"
        / "competitions"
        / "diabetic-retinopathy-detection"
    )


def split_training_labels(
    labels: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    validation_indices = []
    for level, group in labels.groupby("level", sort=True):
        if len(group) < 2:
            continue
        validation_count = round(len(group) * validation_fraction)
        validation_count = min(max(validation_count, 1), len(group) - 1)
        validation_indices.extend(
            group.sample(
                n=validation_count,
                random_state=random_seed + int(level),
            ).index
        )

    validation = labels.loc[validation_indices].sort_index().copy()
    training = labels.drop(index=validation_indices).copy()
    return training, validation


def print_labeled_set(name: str, dataset: pd.DataFrame) -> None:
    print(f"\n--- {name} set ---")
    print(f"Images: {len(dataset):,}")
    print("\nRetinopathy grade counts:")
    print(dataset["level"].value_counts().sort_index().to_string())
    print("\nFirst 5 rows:")
    print(dataset[["image", "level"]].head().to_string(index=False))


def print_archive_parts(data_dir: Path, dataset_name: str) -> None:
    parts = sorted(data_dir.glob(f"{dataset_name}.zip.*"))
    if not parts:
        raise FileNotFoundError(
            f"No {dataset_name} image archive parts found in: {data_dir}"
        )

    total_bytes = sum(path.stat().st_size for path in parts)
    print(
        f"\n{dataset_name.title()} image archive: {len(parts)} parts, "
        f"{total_bytes:,} bytes total"
    )
    for path in parts:
        print(f"  {path.name}: {path.stat().st_size:,} bytes")


def load_dataset_metadata(
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or get_data_dir()
    labels_archive = data_dir / "trainLabels.csv.zip"
    submission_archive = data_dir / "sampleSubmission.csv.zip"

    for path in (labels_archive, submission_archive):
        if not path.is_file():
            raise FileNotFoundError(f"Required competition archive not found: {path}")

    print(f"Competition data: {data_dir}")
    print_archive_parts(data_dir, "train")
    print_archive_parts(data_dir, "test")

    with ZipFile(labels_archive) as archive:
        with archive.open("trainLabels.csv") as labels_file:
            labels = pd.read_csv(labels_file)

    required_label_columns = {"image", "level"}
    if not required_label_columns.issubset(labels.columns):
        raise ValueError(
            f"Training labels must contain columns: {required_label_columns}"
        )

    training, validation = split_training_labels(labels)
    print(f"\nLabeled training images: {len(labels):,}")
    print("Validation set: reproducible, stratified 20% split from training labels.")
    print_labeled_set("Train", training)
    print_labeled_set("Validation", validation)

    with ZipFile(submission_archive) as archive:
        with archive.open("sampleSubmission.csv") as submission_file:
            submission = pd.read_csv(submission_file)

    if "image" not in submission.columns:
        raise ValueError("Test submission template must contain an 'image' column.")
    test_images = submission[["image"]].copy()

    print("\n--- Test set ---")
    print(f"Images: {len(test_images):,}")
    print("Labels: not provided for the competition test set.")
    print("First 5 image IDs:")
    print(test_images.head().to_string(index=False))
    print(
        "Note: sampleSubmission.csv contains placeholder levels, "
        "not ground-truth test labels."
    )

    return training, validation, test_images


def main() -> None:
    print("===== Diabetic Retinopathy Dataset Metadata =====")
    load_dataset_metadata()


if __name__ == "__main__":
    main()
