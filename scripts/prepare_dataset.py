import pandas as pd
import re

input_path = "data/imdb_raw.csv"  # rename your IMDB file to imdb_raw.csv
output_path = "data/train.csv"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text)  # remove <br> tags
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text


print(" Loading dataset...")
df = pd.read_csv(input_path)

# Rename columns to match your training code
df = df.rename(columns={"review": "text", "sentiment": "label"})

# Convert labels to integers
df["label"] = df["label"].map({"negative": 0, "positive": 1})

# Clean text
df["text"] = df["text"].apply(clean_text)

# Drop empty rows
df = df[df["text"].str.strip() != ""]

print(" Cleaned dataset:")
print(df.head())

df.to_csv(output_path, index=False)
print(f"\n Preprocessed dataset saved → {output_path}")
print(" Total rows:", len(df))
