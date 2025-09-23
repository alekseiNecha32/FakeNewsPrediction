import pandas as pd



def load_combined_news():

    df_fake = pd.read_csv("data/Fake.csv")

    df_true = pd.read_csv("data/True.csv")



    df_fake["label"] = 0

    df_true["label"] = 1



    df = pd.concat([df_fake, df_true], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)



    print(df.head())  # optional preview

    return df