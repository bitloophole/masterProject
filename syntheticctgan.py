import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# =====================================================
# CONFIG
# =====================================================

DATA_PATH = "iot_dataset_after_undersampling.csv"

OUTPUT_SYNTHETIC = "zero_day_like_synthetic_100k.csv"
OUTPUT_TSNE = "tsne_zero_day_like_synthetic.png"

LABEL_COL = "Attack_Category"
DROP_COLS = ["Label"]          # multi-label column if exists

NORMAL_LABEL = 0
ATTACK_LABEL = 1

TOTAL_SYNTHETIC = 100000
SYN_NORMAL = TOTAL_SYNTHETIC // 2
SYN_ATTACK = TOTAL_SYNTHETIC // 2

CTGAN_EPOCHS = 500
RANDOM_STATE = 42

# deviation strength
# increase to 0.30 for more separation
# decrease to 0.10 for more realistic overlap
DEVIATION_STRENGTH_NORMAL = 0.15
DEVIATION_STRENGTH_ATTACK = 0.15

TSNE_SAMPLE_PER_GROUP = 20000

np.random.seed(RANDOM_STATE)

# =====================================================
# 1. LOAD REAL DATA
# =====================================================

df = pd.read_csv(DATA_PATH)
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# Drop multi-label or extra label column if present
for col in DROP_COLS:
    if col in df.columns:
        df = df.drop(columns=[col])

# Convert Attack_Category to binary if needed
if df[LABEL_COL].dtype == "object":
    df[LABEL_COL] = df[LABEL_COL].apply(
        lambda x: 0 if str(x).strip().upper() in ["BENIGN", "NORMAL"] else 1
    )

df[LABEL_COL] = df[LABEL_COL].astype(int)

# Keep only numeric features + label
df = df.select_dtypes(include=[np.number])

feature_cols = [col for col in df.columns if col != LABEL_COL]

print("Dataset shape:", df.shape)
print("Class distribution:")
print(df[LABEL_COL].value_counts())

# =====================================================
# 2. TRAIN / TEST SPLIT FOR GAN TRAINING
# =====================================================

train_df, _ = train_test_split(
    df,
    test_size=0.2,
    stratify=df[LABEL_COL],
    random_state=RANDOM_STATE
)

normal_train = train_df[train_df[LABEL_COL] == NORMAL_LABEL][feature_cols].reset_index(drop=True)
attack_train = train_df[train_df[LABEL_COL] == ATTACK_LABEL][feature_cols].reset_index(drop=True)

print("\nGAN training data:")
print("Normal:", len(normal_train))
print("Attack:", len(attack_train))

# =====================================================
# 3. TRAIN CTGAN MODELS
# =====================================================

normal_metadata = SingleTableMetadata()
normal_metadata.detect_from_dataframe(normal_train)

attack_metadata = SingleTableMetadata()
attack_metadata.detect_from_dataframe(attack_train)

normal_ctgan = CTGANSynthesizer(
    metadata=normal_metadata,
    epochs=CTGAN_EPOCHS,
    verbose=True
)

attack_ctgan = CTGANSynthesizer(
    metadata=attack_metadata,
    epochs=CTGAN_EPOCHS,
    verbose=True
)

print("\nTraining CTGAN for NORMAL data...")
normal_ctgan.fit(normal_train)

print("\nTraining CTGAN for ATTACK data...")
attack_ctgan.fit(attack_train)

# =====================================================
# 4. GENERATE SYNTHETIC DATA
# =====================================================

print("\nGenerating synthetic normal samples...")
syn_normal = normal_ctgan.sample(num_rows=SYN_NORMAL)
syn_normal = syn_normal[feature_cols].copy()
syn_normal[LABEL_COL] = NORMAL_LABEL

print("\nGenerating synthetic attack samples...")
syn_attack = attack_ctgan.sample(num_rows=SYN_ATTACK)
syn_attack = syn_attack[feature_cols].copy()
syn_attack[LABEL_COL] = ATTACK_LABEL

# =====================================================
# 5. ADD CONTROLLED DEVIATION FOR ZERO-DAY-LIKE BEHAVIOR
# =====================================================

def add_zero_day_like_deviation(synthetic_df, real_reference_df, strength):
    synthetic_df = synthetic_df.copy()

    numeric_cols = [col for col in synthetic_df.columns if col != LABEL_COL]

    real_std = real_reference_df[numeric_cols].std().replace(0, 1)
    real_mean = real_reference_df[numeric_cols].mean()

    noise = np.random.normal(
        loc=0,
        scale=strength,
        size=synthetic_df[numeric_cols].shape
    )

    synthetic_df[numeric_cols] = synthetic_df[numeric_cols] + noise * real_std.values

    # Slight directional shift away from real mean
    direction = np.sign(synthetic_df[numeric_cols] - real_mean)
    synthetic_df[numeric_cols] = synthetic_df[numeric_cols] + (
        direction * strength * 0.5 * real_std.values
    )

    return synthetic_df

syn_normal = add_zero_day_like_deviation(
    syn_normal,
    normal_train,
    DEVIATION_STRENGTH_NORMAL
)

syn_attack = add_zero_day_like_deviation(
    syn_attack,
    attack_train,
    DEVIATION_STRENGTH_ATTACK
)

synthetic_df = pd.concat([syn_normal, syn_attack], ignore_index=True)
synthetic_df = synthetic_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

synthetic_df.to_csv(OUTPUT_SYNTHETIC, index=False)

print("\nSaved synthetic zero-day-like data:", OUTPUT_SYNTHETIC)
print(synthetic_df[LABEL_COL].value_counts())

# =====================================================
# 6. PREPARE DATA FOR t-SNE
# =====================================================

real_normal = df[df[LABEL_COL] == NORMAL_LABEL].copy()
real_attack = df[df[LABEL_COL] == ATTACK_LABEL].copy()

synthetic_normal = synthetic_df[synthetic_df[LABEL_COL] == NORMAL_LABEL].copy()
synthetic_attack = synthetic_df[synthetic_df[LABEL_COL] == ATTACK_LABEL].copy()

real_normal = real_normal.sample(
    n=min(TSNE_SAMPLE_PER_GROUP, len(real_normal)),
    random_state=RANDOM_STATE
)

real_attack = real_attack.sample(
    n=min(TSNE_SAMPLE_PER_GROUP, len(real_attack)),
    random_state=RANDOM_STATE
)

synthetic_normal = synthetic_normal.sample(
    n=min(TSNE_SAMPLE_PER_GROUP, len(synthetic_normal)),
    random_state=RANDOM_STATE
)

synthetic_attack = synthetic_attack.sample(
    n=min(TSNE_SAMPLE_PER_GROUP, len(synthetic_attack)),
    random_state=RANDOM_STATE
)

real_normal["Group"] = "Real Normal"
real_attack["Group"] = "Real Attack"
synthetic_normal["Group"] = "Synthetic Normal"
synthetic_attack["Group"] = "Synthetic Attack"

combined = pd.concat(
    [real_normal, real_attack, synthetic_normal, synthetic_attack],
    ignore_index=True
)

X = combined.drop(columns=[LABEL_COL, "Group"], errors="ignore")
groups = combined["Group"]

X = X.select_dtypes(include=[np.number])
X = X.fillna(X.median())

# =====================================================
# 7. SCALE + PCA + t-SNE
# =====================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=min(20, X_scaled.shape[1]), random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=3000,
    learning_rate="auto",
    init="pca",
    random_state=RANDOM_STATE
)

print("\nRunning t-SNE...")
X_tsne = tsne.fit_transform(X_pca)

tsne_df = pd.DataFrame({
    "TSNE-1": X_tsne[:, 0],
    "TSNE-2": X_tsne[:, 1],
    "Group": groups.values
})

# =====================================================
# 8. PLOT
# =====================================================

plt.figure(figsize=(10, 8))

for group_name in tsne_df["Group"].unique():
    subset = tsne_df[tsne_df["Group"] == group_name]
    plt.scatter(
        subset["TSNE-1"],
        subset["TSNE-2"],
        label=group_name,
        s=5,
        alpha=0.55
    )

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Real and Zero-Day-Like Synthetic IoT Data")
plt.legend(markerscale=3)
plt.tight_layout()

plt.savefig(OUTPUT_TSNE, dpi=300)
plt.show()

print("\nSaved t-SNE plot:", OUTPUT_TSNE)