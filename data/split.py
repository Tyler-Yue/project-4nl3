import pandas as pd

# Load the dataset
df = pd.read_csv("train.csv")

# Create train_label.csv (only class index)
df["class_index"].to_csv("training_label.csv", index=False)

# Create train_data.csv (question_title + question_content)
train_data = df[["question_title", "question_content"]]
train_data.to_csv("training_data.csv", index=False)