import pandas as pd
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# ------------------------------
# CONFIG
# ------------------------------
DATA_PATH = "iot_dataset_after_undersampling.csv"
OUTPUT_PATH = "synthetic_test_setattack.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CTGAN_EPOCHS = 300
CTGAN_VERBOSE = True

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(DATA_PATH)

df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# ------------------------------
# CONVERT TARGET SAFELY
# ------------------------------
print("Original Attack_Category values:")
print(df["Attack_Category"].value_counts())

if df["Attack_Category"].dtype == "object":
    df["Attack_Category"] = df["Attack_Category"].apply(
        lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1
    )
else:
    df["Attack_Category"] = df["Attack_Category"].astype(int)

print("\nAfter conversion:")
print(df["Attack_Category"].value_counts())

# Drop Label column if it exists
if "Label" in df.columns:
    df = df.drop(columns=["Label"])

feature_columns = [col for col in df.columns if col != "Attack_Category"]

# ------------------------------
# TRAIN / TEST SPLIT
# ------------------------------
train_df, real_test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["Attack_Category"],
)

train_df = train_df.reset_index(drop=True)
real_test_df = real_test_df.reset_index(drop=True)

print("\nReal test set distribution:")
print(real_test_df["Attack_Category"].value_counts())

# ------------------------------
# TRAIN CLASS-WISE CTGAN ON TRAINING DATA ONLY
# ------------------------------
benign_train_df = train_df[train_df["Attack_Category"] == 0][feature_columns].reset_index(drop=True)
attack_train_df = train_df[train_df["Attack_Category"] == 1][feature_columns].reset_index(drop=True)

print("\nGAN training data:")
print("Benign training rows:", len(benign_train_df))
print("Attack training rows:", len(attack_train_df))

if len(benign_train_df) == 0:
    raise ValueError("No benign samples found for GAN training.")

if len(attack_train_df) == 0:
    raise ValueError("No attack samples found for GAN training.")

# Metadata
benign_metadata = SingleTableMetadata()
benign_metadata.detect_from_dataframe(benign_train_df)

attack_metadata = SingleTableMetadata()
attack_metadata.detect_from_dataframe(attack_train_df)

# CTGAN models
benign_ctgan = CTGANSynthesizer(
    metadata=benign_metadata,
    epochs=CTGAN_EPOCHS,
    verbose=CTGAN_VERBOSE,
)

attack_ctgan = CTGANSynthesizer(
    metadata=attack_metadata,
    epochs=CTGAN_EPOCHS,
    verbose=CTGAN_VERBOSE,
)

print("\nTraining CTGAN for BENIGN samples...")
#benign_ctgan.fit(benign_train_df)

print("\nTraining CTGAN for ATTACK samples...")
attack_ctgan.fit(attack_train_df)

# ------------------------------
# GENERATE SYNTHETIC TEST SET
# Same size and same class distribution as real test set
# ------------------------------
#real_benign_count = int((real_test_df["Attack_Category"] == 0).sum())
#real_attack_count = int((real_test_df["Attack_Category"] == 1).sum())
NUM_ATTACK_SAMPLES = 300000
NUM_BENIGN_SAMPLES = 0  # optional (set to 0 if you want attack-only)

real_attack_count = NUM_ATTACK_SAMPLES
real_benign_count = NUM_BENIGN_SAMPLES
print("\nGenerating synthetic test set:")
print("Synthetic benign rows:", real_benign_count)
print("Synthetic attack rows:", real_attack_count)

syn_benign = benign_ctgan.sample(num_rows=real_benign_count)
syn_benign = syn_benign[feature_columns].copy()
syn_benign["Attack_Category"] = 0

syn_attack = attack_ctgan.sample(num_rows=real_attack_count)
syn_attack = syn_attack[feature_columns].copy()
syn_attack["Attack_Category"] = 1

synthetic_test_df = pd.concat(
    [syn_benign, syn_attack],
    axis=0,
    ignore_index=True,
)

synthetic_test_df = synthetic_test_df.sample(
    frac=1,
    random_state=RANDOM_STATE,
).reset_index(drop=True)

# ------------------------------
# SAVE SYNTHETIC TEST SET
# ------------------------------
synthetic_test_df.to_csv(OUTPUT_PATH, index=False)

print("\nSynthetic test set saved to:", OUTPUT_PATH)
print("\nSynthetic test set distribution:")
print(synthetic_test_df["Attack_Category"].value_counts())

print("\nTotal synthetic samples:", len(synthetic_test_df))