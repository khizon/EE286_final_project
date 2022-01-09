import pandas as pd
import numpy as np

import os
import gdown

from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torchaudio

if __name__ == '__main__':
    
    if not os.path.exists(os.path.join('data')):
        os.makedirs(os.path.join('data'))
        
    os.system('gdown https://drive.google.com/uc?id=1_IAWexEWpH-ly_JaA5EGfZDp-_3flkN1')
    os.system('unzip -q aesdd.zip -d data/')
    os.rename(os.path.join('data', 'Acted Emotional Speech Dynamic Database'),
              os.path.join('data', 'aesdd'))
    
    data = []
    # Load the annotations file
    for path in tqdm(Path("data/aesdd").glob("**/*.wav")):
        name = str(path).split("/")[-1]
        label = str(path).split('/')[-2]
        path = os.path.join("data", "aesdd", label, name)
        print(path)

        try:
            # There are some broken files
            s = torchaudio.load(path)
            print(s)
            data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            # print(str(path), e)
            pass

        
        
    df = pd.DataFrame(data)
    print(df.head())
    
    # Filter broken and non-existed paths

    print(f"Step 0: {len(df)}")

    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")

    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    
    # Train test split
    save_path = "data"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=101, stratify=df["emotion"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)


    print(train_df.shape)
    print(test_df.shape)