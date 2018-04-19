import pandas as pd


fields=['subject', 'can', 'will', 'one', 'writes', 'article', 'like', 'dont', 'just', 'know', 'get', 'people', 'think', 'also', 'use', 'time', 'good', 'well', 'even', 'new', 'now', 'way', 'god', 'much', 'see', 'first', 'anyone', 'many', 'make', 'say', 'two', 'may', 'right', 'want', 'said', 'really', 'government', 'need', 'windows', 'work', 'thanks', 'file', 'believe', 'system', 'since', 'something', 'problem', 'years', 'ive', 'game', 'might', 'help', 'using', 'used', 'point', 'email', 'please', 'space', 'still', 'jesus', 'team', 'things', 'car', 'drive', 'never', 'last', 'take', 'program', 'key', 'fact', 'back', 'christian', 'going', 'israel', 'image', 'made', 'year', 'gun', 'must', 'armenian', 'state', 'encryption', 'games', 'clipper', 'window', 'law', 'turkish', 'another', 'chip', 'come', 'armenians', 'bible', 'files', 'win', 'jews', 'world', 'read', 'sale', 'dos', 'players','mainCATEGORIES']
df = pd.read_csv("I:\\Masters\\SPRING 18\\SPATIAL AND TEMPORAL\\spatial\\homework 2\\before feature sel\\KNN-Algorithm\\data.csv",usecols=fields)
df.to_csv('checkfinal.csv',encoding = "ISO-8859-1")
#df.nocat= pd.DataFrame.drop(labels='mainCATEGORIES')
print(df)