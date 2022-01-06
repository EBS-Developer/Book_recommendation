import uvicorn
from fastapi import FastAPI,Response
# ML pkg
import joblib,os, pickle
import numpy as np


#Vectorizer
# rating_model = open('models/model_by_rating.pkl', "rb")
model_by_rating = pickle.load(open('models/model_by_rating', 'rb'))
model_by_age = pickle.load(open('models/model_by_age', 'rb'))
book_pivot_rating = pickle.load(open('models/book_pivot_rating', 'rb'))
book_pivot_age = pickle.load(open('models/book_pivot_age', 'rb'))
#init app
app = FastAPI()

@app.get('/')
async def index():
    return {"text": 'Hello API Lovers'}

@app.get('/items/{name}')
async def get_items(name):
    return {"name": name}

#ML Aspect

@app.get('/predict/{book_name}')
async def predict(book_name):
    book_id = np.where(book_pivot_rating.index==book_name)[0][0]
    distances,suggestions=model_by_rating.kneighbors(book_pivot_rating.iloc[book_id,:].values.reshape(1,-1))
    for i in range(len(suggestions)):
        books_by_rating= list(book_pivot_rating.index[suggestions[i]])

    book_id = np.where(book_pivot_age.index==book_name)[0][0]
    distances,suggestions = model_by_age.kneighbors(book_pivot_age.iloc[book_id,:].values.reshape(1,-1))
    for i in range(len(suggestions)):
        books_by_age= list(book_pivot_age.index[suggestions[i]])  

    books = set(books_by_rating + books_by_age) #unique book
    books = list(books)
    books.remove(book_name)
    return {"id":books}
      
    
    # for i in range(len(suggestions)):
    #   return  {"books": np.array(book_pivot_rating.index[suggestions[i]])}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1",port=8000)
