import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_dir, csv_path):
    df = pd.read_csv(csv_path, names=['image_path', 'label'], converters={'image_path': lambda x: f"{data_dir}/{x}.jpeg"})
    df = df[df['image_path'].apply(lambda x: os.path.exists(x))]
    return df

def main(data_dir, csv_path, train_csv_path, val_csv_path, test_csv_path,
         val_size=0.1, test_size=0.1, random_state=42):
    # Load data from CSV
    df = load_data(data_dir, csv_path)

    # First split: separate out the test set
    df_trainval, df_test = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )

    # Second split: separate train and validation from the remaining data
    # val_size is relative to the original dataset, so rescale for the remaining portion
    val_fraction = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_fraction, stratify=df_trainval['label'], random_state=random_state
    )

    # Save the splits to CSV files
    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)} "
          f"(total: {len(df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, validation, and test sets.")
    parser.add_argument("--data_dir", type=str, default="data/diabetic-retinopathy-dataset/resized/train", help="Directory containing images.")
    parser.add_argument("--csv_path", type=str, default="data/diabetic-retinopathy-dataset/trainLabels.csv", help="Path to CSV file containing image labels.")
    parser.add_argument("--train_csv_path", type=str, default="data/diabetic-retinopathy-dataset/train.csv", help="Path to save train CSV file.")
    parser.add_argument("--val_csv_path", type=str, default="data/diabetic-retinopathy-dataset/val.csv", help="Path to save validation CSV file.")
    parser.add_argument("--test_csv_path", type=str, default="data/diabetic-retinopathy-dataset/test.csv", help="Path to save test CSV file.")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of the dataset for the validation split.")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset for the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Seed for random number generator.")

    args = parser.parse_args()

    main(args.data_dir, args.csv_path, args.train_csv_path, args.val_csv_path,
         args.test_csv_path, args.val_size, args.test_size, args.random_state)
