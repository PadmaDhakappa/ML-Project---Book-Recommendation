
import numpy as np
import pandas as pd
import os
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import time
from matplotlib import pyplot as plt

import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
import plotly.express as px
from PIL import Image
import requests
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore")


books = pd.read_csv("data/"+'Books.csv')
users = pd.read_csv("data/"+'Users.csv')
ratings = pd.read_csv("data/"+'Ratings.csv')

books['Year-Of-Publication'] = pd.to_numeric(books['Year-Of-Publication'], errors='coerce')
books['Year-Of-Publication'].fillna(2000, inplace=True)
books['Year-Of-Publication'] = books['Year-Of-Publication'].astype('int32')
books['Image-URL-M'] = books['Image-URL-M'].str.replace('http://', 'https://')

combined_df = pd.merge(books, ratings, on='ISBN')
combined_df.dropna(inplace=True)

combined_df = pd.merge(combined_df, users, on='User-ID')
df = combined_df[combined_df['Book-Rating'] != 0]
# combined_df = combined_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

new_df=df[df['User-ID'].map(df['User-ID'].value_counts()) > 100]
users_matrix=new_df.pivot_table(index=["User-ID"],columns=["Book-Title"],values="Book-Rating")
users_matrix.fillna(0, inplace=True)


def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))

def top_books(df):
    books_top20 = df['Book-Title'].value_counts().head(20)
    books_top20 = list(books_top20.index)

    top20_books = pd.DataFrame(columns=df.columns)

    for book in books_top20:
        cond_df = df[df['Book-Title'] == book]

        top20_books = pd.concat([top20_books, cond_df], axis=0)

    top20_books = top20_books[top20_books['Book-Rating'] != 0]
    top20_books = top20_books.groupby('Book-Title')['Book-Rating'].agg('mean').reset_index().sort_values(
        by='Book-Rating', ascending=False)

    return top20_books

top20_books = top_books(df)
top10_books = top20_books.head(10)

def popular_books2():
    with open("popular_df_5.pkl", "rb") as file:
        # Load the object from the file
        popular_df = pickle.load(file)
    recommended_books = []
    for index, row in popular_df.iterrows():
        book = row['Book Title']
        author = row['Author']
        URL = row['Image-URL-M']
        recommended_books.append((book, author, URL))
    return recommended_books
def popular_books():

    for (book, ratings) in zip(top10_books['Book-Title'], top10_books['Book-Rating']):
        prGreen(book)
        print("Rating",end='->')
        prRed(round(ratings,1))
        print("-"*50)
def content_based(book_title):

    book_title = str(book_title)
    recommended_books = []
    if book_title in combined_df['Book-Title'].values:
        count_rate = pd.DataFrame(combined_df['Book-Title'].value_counts().rename('Book-Title'))
        rare_books=count_rate[count_rate["Book-Title"]<=200].index
        common_books=combined_df[~combined_df["Book-Title"].isin(rare_books)]

        if book_title in rare_books:

            recommended_books = popular_books2()
        else:
            common_books=common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)

            common_books["index"]=[i for i in range(common_books.shape[0])]
            common_books['Book-Title'] = common_books['Book-Title'].astype('object')
            common_books['Book-Author'] = common_books['Book-Author'].astype('object')
            common_books['Publisher'] = common_books['Publisher'].astype('object')

            targets=["Book-Title","Book-Author","Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in range(common_books[targets].shape[0])]

            vectorizer=CountVectorizer()
            common_booksVector=vectorizer.fit_transform(common_books["all_features"])

            similarity=cosine_similarity(common_booksVector)
            index=common_books[common_books["Book-Title"]==book_title]["index"].values[0]
            similar_books=list(enumerate(similarity[index]))
            similar_booksSorted=sorted(similar_books,key=lambda x:x[1],reverse=True)[1:6]
            r_books=[]

            for i in range(len(similar_booksSorted)):
                r_books.append(common_books[common_books["index"]==similar_booksSorted[i][0]]["Book-Title"].item())

            prYellow(f"Recommend Books similar to {book_title}:\n")
            for book in r_books:
                prGreen(book)
                print("Rating",end='->')
                prRed(round(df[df['Book-Title'] == book]['Book-Rating'].mean(), 2))

                recommended_books.append((book, df[df['Book-Title'] == book]['Book-Author'].iloc[0],
                                          df[df['Book-Title'] == book]['Image-URL-M'].iloc[0]))

            print(recommended_books)
        return recommended_books
    else:
        prYellow("This book is not in our library, check out our most popular books:")
        # print()
        return popular_books2()

def user_based_coll_rs(user_id):

    users_fav=new_df[new_df["User-ID"]==user_id].sort_values(["Book-Rating"],ascending=False)[0:5]

    prYellow("Ypur Top Favorite books: \n")

    for book in users_fav['Book-Title']:

        prGreen(book)
        print("Rating", end='->')
        prRed(round(df[df['Book-Title'] == book]['Book-Rating'].mean(), 2))
        print("-"*50)

    print("\n\n")

    index=np.where(users_matrix.index==2033)[0][0]
    similarity=cosine_similarity(users_matrix)
    similar_users = list(enumerate(similarity[index]))
    similar_users = sorted(similar_users,key = lambda x:x[1],reverse=True)[0:5]

    users_id=[]

    for i in similar_users:

            data=df[df["User-ID"]==users_matrix.index[i[0]]]
            users_id.extend(list(data.drop_duplicates("User-ID")["User-ID"].values))


    x=new_df[new_df["User-ID"]==user_id]
    recommend_books=[]
    user=list(users_id)

    for i in user:

        y=new_df[(new_df["User-ID"]==i)]
        sim_books=y.loc[~y["Book-Title"].isin(x["Book-Title"]),:]
        sim_books=sim_books.sort_values(["Book-Rating"],ascending=False)[0:5]
        recommend_books.extend(sim_books["Book-Title"].values)


    prYellow("Recommended for you: \n")
    recommended_books_x = []
    for book in recommend_books:
        prGreen(book)
        print("Rating",end='->')
        prRed(round(df[df['Book-Title'] == book]['Book-Rating'].mean(),2))
        print("-"*50)
        recommended_books_x.append((book, df[df['Book-Title'] == book]['Book-Author'].iloc[0],
                                  df[df['Book-Title'] == book]['Image-URL-M'].iloc[0]))

    return recommended_books_x

def item_based_coll_rs(book_title):

    book_title = str(book_title)
    if book_title in combined_df['Book-Title'].values:

        count_rate = pd.DataFrame(df['Book-Title'].value_counts().rename('Book-Title'))

        rare_books=count_rate[count_rate["Book-Title"]<=100].index

        common_books=df[~df["Book-Title"].isin(rare_books)]
        recommended_books = []
        if book_title in rare_books:
            prYellow("A rare book, so u may try our popular books: \n ")
            recommended_books = popular_books2()

        else:

            item_based_cb = common_books.pivot_table(index=["User-ID"],columns=["Book-Title"],values="Book-Rating")
            sim = item_based_cb[book_title]
            recommendation_df=pd.DataFrame(item_based_cb.corrwith(sim).sort_values(ascending=False)).reset_index(drop=False)

            if not recommendation_df['Book-Title'][recommendation_df['Book-Title'] == book_title].empty:
                recommendation_df=recommendation_df.drop(recommendation_df[recommendation_df["Book-Title"]==book_title].index[0])

            less_rating=[]
            for i in recommendation_df["Book-Title"]:
                if df[df["Book-Title"]==i]["Book-Rating"].mean() < 5:
                    less_rating.append(i)

            if recommendation_df.shape[0] - len(less_rating) > 5:

                recommendation_df=recommendation_df[~recommendation_df["Book-Title"].isin(less_rating)]
                recommendation_df.columns=["Book-Title","Correlation"]


            for (candidate_book, corr) in zip(recommendation_df['Book-Title'], recommendation_df['Correlation']):
                corr_thershold = 0.7
                if corr > corr_thershold:
                    ratings = df[df['Book-Title'] == candidate_book]['Book-Rating'].mean()
                    prGreen(candidate_book)
                    print("Rating ", end = '->')
                    prRed(round(ratings,1))
                    print("-"*50)

                    recommended_books.append((candidate_book, df[df['Book-Title'] == candidate_book]['Book-Author'].iloc[0],
                                              df[df['Book-Title'] == candidate_book]['Image-URL-M'].iloc[0]))
                else:
                    break
        return recommended_books

    else:
        prYellow("This book is not in our library, check out our most popular books:")
        print()
        return popular_books2()