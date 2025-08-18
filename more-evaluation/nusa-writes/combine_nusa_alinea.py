import pandas as pd
import os
import glob
import re
import regex

def combine_nusa_alinea_datasets():
    languages = ['jav', 'sun', 'mad', 'bug', 'btk']
    dataset_types = ['emot', 'paragraph', 'topic']
    base_dir = 'more-evaluation/nusa-writes'
    all_data = []
    
    for lang in languages:
        for dataset_type in dataset_types:
            pattern = f'nusa_alinea-{dataset_type}-{lang}-*.csv'
            files = glob.glob(os.path.join(base_dir, pattern))
            
            print(f"Processing {dataset_type} for language {lang}: {len(files)} files found")
            
            for file in files:
                try:
                    df = pd.read_csv(file)
                    df['label'] = lang
                    # Change 'btk' to 'bbc' in the language column
                    df['label'] = df['label'].replace('btk', 'bbc')
                    df = df[['text', 'label']]
                    all_data.append(df)
                    print(f"  - Loaded {len(df)} samples from {os.path.basename(file)}")
                except Exception as e:
                    print(f"  - Error reading {file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['text'])
        output_file = 'nusa_alinea_combined_dataset.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"\nCombined dataset saved to: {output_file}")
        print(f"Total samples: {len(combined_df)}")
        print(f"Language distribution:")
        # Language distribution:
        # language
        # jav    10188
        # sun     9594
        # mad     5178
        # bbc     4888
        # bug     1000
        print(combined_df['label'].value_counts())
        
        print(f"\nSample of combined dataset:")
        print(combined_df.head(10))
        
        return combined_df
    else:
        print("No data found!")
        return None

def remove_newlines(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\n", "")

def remove_parentheses(text: str) -> str:
    """
    Removes any content inside () including the parentheses.
    """
    return re.sub(r'\([^)]*\)', '', text)

def remove_equals(text: str) -> str:
    return text.replace("==", "")

def remove_doubledashes(text: str) -> str:
    return text.replace("--", "")

def remove_links(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # matches http:// or https:// followed by non-space characters
    return re.sub(r"http\S+|www\S+", "", text)

def keep_latin_only(text: str) -> str:
    """
    Keeps only Latin script characters (with diacritics),
    spaces, and common punctuation. Removes all non-Latin scripts.
    """
    # \p{Latin} matches all Latin characters including diacritics
    return regex.sub(r"[^\p{Latin}\s.,;:!?'\-–—()/]", "", text)

def parse_dataset():
    df_all = pd.read_csv("nusa_alinea_combined_dataset.csv")
    df_parsed = df_all[["label"]].copy()
    df_parsed["text"] = (
        df_all["text"]
        .apply(lambda x: remove_newlines(x))
        .apply(lambda x: remove_parentheses(x))
        .apply(lambda x: remove_equals(x))
        .apply(lambda x: remove_doubledashes(x))
        .apply(lambda x: remove_links(x))
        .apply(lambda x: keep_latin_only(x))
    )
    
    df_parsed = df_parsed.dropna()
    df_parsed = df_parsed[df_parsed['text'].str.strip() != '']
    df_parsed = df_parsed.drop_duplicates(subset=['text'])
    df_parsed = df_parsed.reset_index(drop=True)
    
    df_parsed.to_csv("nusa_alinea_combined_dataset_parsed.csv", index=False)
    print(f"Total samples: {len(df_parsed)}")
    print(f"Language distribution:")
    print(df_parsed['label'].value_counts())
    print(f"\nSample of combined dataset:")
    print(df_parsed.head(10))

if __name__ == "__main__":
    # combine_nusa_alinea_datasets()
    parse_dataset()