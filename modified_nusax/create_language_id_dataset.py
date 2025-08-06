import os
import pandas as pd

# Get the base directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../nusax-main/datasets'))
mt_dir = os.path.join(base_dir, "mt")
sentiment_dir = os.path.join(base_dir, "sentiment")
out_dir = script_dir

# Languages to include and their mapping to folder/column names
languages = [
    ("english", "en"),
    ("indonesian", "ind"),
    ("javanese", "jav"),
    ("sundanese", "sun"),
    ("balinese", "ban"),
    ("toba_batak", "bbc"),
    ("madurese", "mad"),
    ("acehnese", "ace"),
    ("buginese", "bug"),
]

mt_col_map = {
    "english": "english",
    "indonesian": "indonesian",
    "javanese": "javanese",
    "sundanese": "sundanese",
    "balinese": "balinese",
    "toba_batak": "toba_batak",
    "madurese": "madurese",
    "acehnese": "acehnese",
    "buginese": "buginese",
}

splits = ["train", "test", "valid"]

rows = []
# --- MT dataset ---
for split in splits:
    mt_path = os.path.join(mt_dir, f"{split}.csv")
    if os.path.exists(mt_path):
        mt_df = pd.read_csv(mt_path)
        for lang_key, lang_label in languages:
            if mt_col_map[lang_key] in mt_df.columns:
                for text in mt_df[mt_col_map[lang_key]].dropna():
                    rows.append({"text": str(text), "label": lang_label})

# --- Sentiment dataset ---
for split in splits:
    for lang_key, lang_label in languages:
        sent_path = os.path.join(sentiment_dir, lang_key, f"{split}.csv")
        if os.path.exists(sent_path):
            sent_df = pd.read_csv(sent_path)
            if "text" in sent_df.columns:
                for text in sent_df["text"].dropna():
                    rows.append({"text": str(text), "label": lang_label})

# Save combined file
out_path = os.path.join(out_dir, "language_id_dataset.csv")
pd.DataFrame(rows).to_csv(out_path, index=False)

print("Language identification dataset created in:", out_path) 