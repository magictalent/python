import os
import pandas as pd

data_dir = ""  # path to your data folder
labels = ["positive", "negative"]

rows = []

for label in labels:
    folder = os.path.join(data_dir, label)
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
                rows.append([text, label])

# Save as CSV
df = pd.DataFrame(rows, columns=["text", "label"])
df.to_csv("sentiment_dataset.csv", index=False, encoding="utf-8")

print("✅ CSV file created: sentiment_dataset.csv")
