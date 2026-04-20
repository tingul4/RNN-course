#%%
import pandas as pd

df = pd.read_csv("dataset/train_v2_drcat_02.csv")  # Example filename
df = df[['text', 'label']] # Ensure columns exist

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("calculating Word Count and Vocabulary Richness...")

# Word Count
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# Vocabulary Richness (Type-Token Ratio - Unique words / Total words)
def calc_richness(text):
    words = str(text).split()
    if len(words) == 0: return 0
    return len(set(words)) / len(words)

df['vocab'] = df['text'].apply(calc_richness)

# Assume 0 = Human, 1 = AI for binary classification
df['source'] = df['label'].map({0: 'Human', 1: 'AI'})

# Summary Statistics
stats = df.groupby('source')[['word_count', 'vocab']].describe()
print("\n=== Statistical Summary (Mean, Std, Min, Max) ===")
print(stats)

# Plotting the distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Word Count
# use 95th percentile to cut off extreme outliers for clearer plot
cutoff = df['word_count'].quantile(0.95)
sns.histplot(data=df[df['word_count'] <= cutoff], x='word_count', hue='source', bins=50, kde=True, ax=axes[0], stat='density', common_norm=False)
axes[0].set_title('Word Count Distribution (Human vs AI)')

# Plot 2: Vocabulary Richness
sns.histplot(data=df, x='vocab', hue='source', bins=50, kde=True, ax=axes[1], stat='density', common_norm=False)
axes[1].set_title('Vocabulary Richness Distribution (Human vs AI)')

plt.tight_layout()
plt.savefig('eda_comparison.png')
print("\n[Done] View 'eda_comparison.png' to see graphical distributions!")
#%%