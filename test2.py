
import pandas as pd
find_txt = "2925760802_50c1e84936.jpg#0"
label_df = pd.read_csv("Data\Flicker8k_text\Flickr8k.lemma.token.txt", sep= "\t", names= ["captionID", "caption"])

found_caption = label_df.loc[label_df["captionID"] == find_txt]
print(found_caption.iloc[0]["caption"])
print(type(found_caption.iloc[0]["caption"]))

