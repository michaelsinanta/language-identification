import pandas as pd
import os

csv_path = os.path.join(os.path.dirname(__file__), 'language_id_dataset.csv')
df = pd.read_csv(csv_path)

label_counts = df['label'].value_counts().sort_index()
total = len(df)

print(total)

print("Language Distribution:")
for label, count in label_counts.items():
    percent = 100 * count / total
    print(f"{label:12}: {count:6} ({percent:.2f}%)")
print(f"\nTotal samples: {total}")

null_label = df['label'].isnull().sum()
null_text = df['text'].isnull().sum()
print(f"\nRows with null label: {null_label}")
print(f"Rows with null text: {null_text}") 