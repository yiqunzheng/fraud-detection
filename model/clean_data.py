import pandas as pd 
import numpy as np 
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


def clean(input_data, isTrain=True, isjson=False):
    if isjson:
        # data = pd.concat((pd.read_json(d) for d in input_data), axis=0)
        data = pd.DataFrame(input_data)
        # data = pd.read_json(input_data)
    else:
        data = pd.read_json(input_data)

    # label fraud
    if isTrain:
        data['fraud'] = data['acct_type'].\
                            map(lambda x: x != 'premium').astype('int')

    # create categoricals for null columns
    null_cols = ['has_header', 'org_twitter', 'venue_country']
    for c in null_cols:
        data[c + '_null'] = data[c].isnull().astype('int')

    # create previous payout categorical feature: if empty list or not
    data['prev_payout_empty'] = \
        data['previous_payouts'].map(lambda x: len(x) == 0)

    # # create categorical whether org_name is empty or not
    # data['org_name_empty'] = \
    #     data['org_name'].map(lambda x: x == '')

    # create whether they sold any ticket
    data['quantity_sold'] = data['ticket_types'].map(map_ticket_total)

    # create time between end and published
    data['time_end_pub'] = data['event_end'] - data['event_published']

    # create time between event created and event published
    data['time_create_pub'] = data['event_created'] - data['event_published']

    # create time between event end and created
    data['time_end_create'] = data['event_end'] - data['event_created']

    # create time between event published and user created
    data['time_pub_user'] = data['event_published'] - data['user_created']


    numeric_cols = []
    string_cols = []
    for c in data.columns:
        if data[c].dtype != 'object':
            numeric_cols.append(c)
        else:
            string_cols.append(c)

    numerics = data[numeric_cols]

    # drop_cols = [
    #              'delivery_method_null',
    #              'org_facebook_null',
    #              'event_published_null',
    #              'country_null',
    #              'sale_duration_null',
    #              'sale_duration_null',
    #              'venue_name_null',
    #              'approx_payout_date',
    #              'event_created',
    #              'event_end',
    #              'event_published',
    #              'user_created'
    #              ]
    # numerics = numerics.drop(drop_cols, axis=1)

    if isTrain:
        numerics.to_csv('data/clean_numeric.csv', index=False)
    else:
        return numerics


def clean_desc(filename, isTrain=True):
    data = pd.read_json(filename)
    if isTrain:
        data['fraud'] = data['acct_type'].\
                            map(lambda x: x != 'premium').astype('int')
    stemmer = SnowballStemmer('english')
    data['content'] = data['description'].map(lambda x: map_desc(x, stemmer))
    if isTrain:
        data[['fraud', 'content']].to_csv('data/desc_content.csv', index=False)
    else:
        return data['content']


def map_desc(html_string, stemmer):
    soup = BeautifulSoup(html_string)
    content = ''
    for s in soup.find_all():
        txt = s.text.lower()
        if txt:
            txt.encode('ascii', 'ignore')
        content += txt

    stemmed = [stemmer.stem(word) for word in word_tokenize(content)]
    return ' '.join(stemmed)

def map_ticket_total(row):
    total = 0
    for d in row:
        total += d['quantity_sold']

    return total == 0

if __name__ == '__main__':
    clean('data/train_new.json')
    # clean_desc('data/train_new.json')