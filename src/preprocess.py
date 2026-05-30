import pandas as pd

def load_and_clean():

    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true])

    df = df.drop_duplicates()

    df["content"] = (
        df["title"] + " " + df["text"]
    )

    return df
