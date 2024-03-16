# import pickle
# pickle.load(open('popular.pkl','rb'))
# from app import recommend

# recommend("The Notebook")

import recommender_system_utilities as recommendation

# recommendation.item_based_coll_rs('Me Talk Pretty One Day')
# recommendation.user_based_coll_rs(2033)
# recommendation.content_based("Clara Callan")
print(recommendation.popular_books2())