from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import recommender_system_utilities as recommendation

pt = pd.read_pickle("pt.pkl")
books = pd.read_pickle("books.pkl")
similarity_scores = pd.read_pickle("similarity_scores.pkl")
popular_df = pd.read_pickle("popular.pkl")

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )


# @app.route('/recommend')
# def recommend_ui():
#     return render_template('recommend.html')

@app.route('/cbrc')
def cbrc_ui():
    return render_template('cbrc.html')

@app.route('/cbrc_item')
def cbrc__item_ui():
    return render_template('cbrc_item.html')

@app.route('/cf')
def cf_ui():
    return render_template('cf.html')

@app.route('/team_introduction')
def team_introduction_ui():
    return render_template('team_introduction.html')


@app.route('/cbrc_item_based',methods=['post'])
def cbrc_item_based():
    user_input = request.form.get('user_input')
    # user_input = user_input.lower()
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    similar_items = recommendation.item_based_coll_rs(user_input)
    data = []
    for i in similar_items:
        item = []
        item.append(i[0])
        item.append(i[1])
        item.append(i[2])
        data.append(item)

    return render_template('cbrc.html',data=data)


@app.route('/cbrc_user_based',methods=['post'])
def cbrc_user_based():
    user_input = request.form.get('user_input')
    # user_input = user_input.lower()
    # index = np.where(pt.index == user_input)[0][0]
    # similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    similar_items = recommendation.user_based_coll_rs(user_input)
    data = []
    for i in similar_items:
        item = []
        item.append(i[0])
        item.append(i[1])
        item.append(i[2])
        data.append(item)

    print(data)

    return render_template('cbrc.html',data=data)

@app.route('/cf_recommend_books',methods=['post'])
def cf():
    user_input = request.form.get('user_input')
    # user_input = user_input.lower()
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    similar_items = recommendation.content_based(user_input)
    data = []
    for i in similar_items:
        item = []
        item.append(i[0])
        item.append(i[1])
        item.append(i[2])
        data.append(item)

    print(data)
    return render_template('cf.html',data=data)


# @app.route('/recommend_books',methods=['post'])
# def recommend():
#
#     user_input = request.form.get('user_input')
#     user_input = user_input.lower()
#     index = np.where(pt.index == user_input)[0][0]
#     similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
#
#     data = []
#     for i in similar_items:
#         item = []
#         temp_df = books[books['Book-Title'] == pt.index[i[0]]]
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
#         item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
#
#         data.append(item)
#
#     print(data)
#
#     return render_template('recommend.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)