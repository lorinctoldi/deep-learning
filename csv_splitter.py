import csv
import os
import random

DATA_ROOT = "./data/"

INPUT_CSV = os.path.join(DATA_ROOT, "segmentations.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV   = os.path.join(DATA_ROOT, "val.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "test.csv")

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-6

rows = []

# Read all rows in RAM – safe since each row is small and unique per image
with open(INPUT_CSV, "r", newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        rows.append(row)

print(f"Total samples: {len(rows)}")

# Shuffle once
random.shuffle(rows)

n = len(rows)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)

train_rows = rows[:n_train]
val_rows   = rows[n_train:n_train+n_val]
test_rows  = rows[n_train+n_val:]

def write_csv(filename, rows_subset):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows_subset:
            writer.writerow(r)

write_csv(TRAIN_CSV, train_rows)
write_csv(VAL_CSV,   val_rows)
write_csv(TEST_CSV,  test_rows)

print("Done! train.csv, val.csv, and test.csv created.")