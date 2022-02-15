import pandas as pd
from langdetect import detect
import re
from googletrans import Translator
#from easynmt import EasyNMT
#model = EasyNMT('opus-mt')
import matplotlib.pyplot as plt






train_data=pd.read_csv("mediaeval-2015-trainingset.txt", sep="	")
test_data=pd.read_csv("mediaeval-2015-testset.txt", sep="	")

translator = Translator()
translation = translator.translate("Der Himmel ist blau und ich mag Bananen", dest='en')
print(translation.text)


# df_train=pd.DataFrame(data=train_data)
df_train=pd.DataFrame(data=test_data)
#print(df_train)

translator = Translator()
translator.raise_Exception = True
translation = model.translate("Der Himmel ist blau und ich mag Bananen", target_lang='en')
print(translation)


#output: 'The sky is blue and I like bananas'
#Get the proportion
#print(df_train["label"].value_counts('fake'))

# Data Characterization
train_data.head()
train_data.info()

test_data.head()
test_data.info()

df_train.rename(columns={'imageId(s)':'img'},inplace=True)
print(len(df_train))

#print(df_train.img.str.split('_').str[0])

img_count=df_train.groupby(df_train.img.str.split('_').str[0])['tweetId'].nunique()
#print(img_count)

selector=[]



for img in df_train['img']:
    if "sandy" in img:
        selector.append(True)
    else:
        selector.append(False)

isEvent=pd.Series(selector)
df_event=df_train[isEvent].head(50)

for tweet in df_event['tweetText']:
    print(tweet)

langs=dict()
for tweet in df_train['tweetText']:
    try:
        lan = detect(tweet)
    except:
        pass
        lan = "unknow"
        print(tweet)

    if lan in langs.keys():
        langs[lan]=langs[lan]+1
    else:
        langs[lan]=1


print(langs.keys())

name=[]
values=[]
for i in langs.keys():
    temp=i+'   '
    name.append(temp)

for j in langs.values():
    temp=j
    values.append(j)
v=[]
n=[]
print(f'length{len(values)}')
for j in range(len(values)):
    if values[j]>=100:
        v.append(values[j])
        n.append(name[j])
plt.barh(n, v)
plt.show()


# data pre-process
# changing humor to fake
df_train.loc[(df_train.label=='humor'),'label'] = 'fake'

#Removing retweets,reposts, and modify tweets

rtPattern1 = "(RT|rt|MT|mt|RP|rp):? @\w*:?"
rtPattern2 = "(\bRT\b|\brt\b|\bMT\b|\bmt\b|\bRP\b|\brp\b)"
rtPattern3 = "(@\w*:)"
rtPattern4 = "(#rt|#RT|#mt|#MT|#rp|#retweet|#Retweet|#modifiedtweet|#modifiedTweet|#ModifiedTweet|#repost|#Repost)"
rtPattern5 = "(via @\w*)"

retweets = df_train['tweetText'].str.contains(rtPattern1)
df_train = df_train[~retweets]
retweets = df_train['tweetText'].str.contains(rtPattern2)
df_train = df_train[~retweets]
retweets = df_train['tweetText'].str.contains(rtPattern3)
df_train = df_train[~retweets]
retweets = df_train['tweetText'].str.contains(rtPattern4)
df_train = df_train[~retweets]
retweets = df_train['tweetText'].str.contains(rtPattern5)
df_train = df_train[~retweets]

#removing @username
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'@\S+', "", text))

#Removing urls
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'http\S+',"",text))

#Removing emojs
emojis = re.compile("["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)


df_train['tweetText'] = df_train['tweetText'].apply(lambda text: emojis.sub(r'', text) if emojis.search(text) else text)

#Removing & and changing lines
df_train['tweetText'] = df_train['tweetText'].apply(lambda text: re.sub(r'\\n|&amp;','',text))


# for tweet in df_train['tweetText'].head(50):
#     print(tweet)
#Translating
print(len(df_train))
TranslateTweet=[]
i=0
errornum=0
for tweet in df_train['tweetText']:
    try:
        lan=detect(tweet)
    except:
        pass
        lan='unknow'
        print(lan)

    #print(i,tweet)
    if lan != 'en' and lan != None and lan != 'unknow':
        # print(i, lan,tweet)
        # translator = Translator()
        # translation = translator.translate("Der Himmel ist blau und ich mag Bananen", dest='en')
        try:
            translator = Translator()
            translation = translator.translate(tweet, dest='en')
            # print(translation.text)
            tweet = translation.text
        except:
            pass
            tweet = tweet
            errornum += 1
            print("can't translate by model num", errornum)
        # TrText=translator.translate(tweet,dest='en')
        print(i,lan,tweet)


    #print(i,tweet)
    #print(i)
    TranslateTweet.append(tweet)
    i += 1
    if i%len(df_train)%1000 == 0:
        print("Translating progerss",i/len(df_train))

Translating={'tweetId':df_train["tweetId"],
    'TweetTrans':TranslateTweet,
    'userId':df_train["userId"],
    'img':df_train["img"],
    'username':df_train["username"],
    'timestamp':df_train["timestamp"],
    'label':df_train["label"]
             }
df_translate=pd.DataFrame(data=Translating)
df_translate.to_csv('./Google Tran Test Tweet.csv')
print('Finished', 'All Num',i ,'Error Num',errornum)
