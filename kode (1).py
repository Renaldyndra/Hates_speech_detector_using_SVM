import pandas as pd
from NLP_Models import TextMining as tm
from NLP_Models import CleanText as ct
from NLP_Models import modelling as mdg
from NLP_Models import deepHateSpechDetection as dhsd


#HATESPEECH
dataPath = tm.crawlFiles('./NLP_Models/data/OFFENSIVE/HateSpeech/HATE/', types = 'json')
data = pd.concat([pd.read_json(f, lines = True) for f in dataPath])
data = data[['created_at', 'date', 'time', \
             'user_id', 'username', 'name',\
                 'tweet','mentions', 'urls', 'photos', 'replies_count', \
                     'retweets_count', 'likes_count', 'hashtags',\
                         'link', 'retweet', 'quote_url',\
                             'near', 'geo', 'source', 'user_rt_id', 'user_rt',\
                                 'retweet_id', 'reply_to', 'retweet_date']]
data['Label'] = '-1'

    
data.to_csv('./NLP_Models/data/dataHate.csv')

#HATESPEECH  
dataPathnhs = tm.crawlFiles('./NLP_Models/data/OFFENSIVE/NonHateSpeech/NON/', types = 'json')
datanhs = pd.concat([pd.read_json(f, lines = True) for f in dataPathnhs])
datanhs = datanhs[['created_at', 'date', 'time', \
             'user_id', 'username', 'name',\
                 'tweet','mentions', 'urls', 'photos', 'replies_count', \
                     'retweets_count', 'likes_count', 'hashtags',\
                         'link', 'retweet', 'quote_url',\
                             'near', 'geo', 'source', 'user_rt_id', 'user_rt',\
                                 'retweet_id', 'reply_to', 'retweet_date']]    
datanhs['Label'] = '1'


dataFinal = pd.concat([data, datanhs])
dataFinal = dataFinal.reset_index(drop= True)    
dataFinal.to_json('./NLP_Models/data/dataFinal.json', orient='records')


#NEWLINE
#data = pd.read_json('./NLP_Models/data/dataFinal.json')
dataFinal.rename(columns={'tweet':'text'}, inplace=True)

dataFinal = ct.cleanningtext(data = dataFinal, both = True, onlyclean = False, sentiment = False)
dataFinal.to_json('./NLP_Models/data/dataFinalClean.json', orient='records')

dataFinal = dataFinal[['text', 'cleaned_text', 'Label']]
modelSVC = mdg.modelling(data = dataFinal, modelname= '202106',\
                         crossval = False,  termfrequency = False, \
                             n_fold = 3, kernel = 'linear', n_jobs=1)
dataFinal.keys()
