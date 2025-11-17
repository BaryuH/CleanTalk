import os
import pandas as pd

os.chdir('.')

INPUT_PATH = 'data/output/submit.csv'
OUTPUT_PATH = 'data/output/final.csv'

df = pd.read_csv(INPUT_PATH)

label_cols = df.columns[-6:]

weights = {}
for power, col in enumerate(reversed(label_cols), start=1):
    weights[col] = 2 ** power

df['points'] = sum(df[col] * w for col, w in weights.items())

threshold_warning = 10
threshold_ban = 50

def classify(points):
    if points >= threshold_ban:
        return "ban"
    elif points >= threshold_warning:
        return "warning"
    else:
        return "safe"

df['decision'] = df['points'].apply(classify)