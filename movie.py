try:
    import pinecone
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer, util
    from sklearn.metrics.pairwise import cosine_similarity
    from annoy import AnnoyIndex
    from pinecone import Pinecone, ServerlessSpec

except Exception as e:
    print("Некоторые модули пропущены :{}".format(e))


pc = Pinecone(api_key="7fc2e4b9-4113-4ce6-b87b-4a7384eb5d65")

index_name = pc.Index("quickstart")
# index_name = "quickstart"

# pc.create_index(
#     name=index_name,
#     dimension=768,
#     metric="cosine",
#     spec=ServerlessSpec(
#         cloud='aws', 
#         region='us-east-1'
#     ) 
# ) 


df=pd.read_csv("Hydra-Movie-Scrape.csv")


"""**Импорт моделей внедрения из ST и загрузка нашего набора данных в режимl**

sentence-transformers— это библиотека, предоставляющая простые методы вычисления вложений 
(плотных векторных представлений) для предложений, абзацев и изображений. Тексты встроены в 
векторное пространство так, что похожий текст находится близко, что позволяет использовать такие 
приложения, как семантический поиск, кластеризацию и извлечение."""

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#q1 = model.encode(df['Summary'].tolist())

"""**После векторизации данных мы можем сохранить их в базе данных pinecone или сохранить их локально.**"""

# Save the vectors to a file
#np.save("vectors1.npy", q1)

"""**С помощью этой команды мы можем напрямую загрузить локальные векторы в программу.**"""

q1 = np.load("vectors1.npy")


def similarity_search_Cos_Sim(inp,k):
    if len(inp) in range(2):
        return "Нет результата"
    query = inp
    query_vector = model.encode([query])
    similarity_scores = cosine_similarity(query_vector.reshape(1, -1),q1).flatten()
    sorted_indices = similarity_scores.argsort()[::-1] #Эта функция используется для получения индекса векторов для доступа к исходным данным.
    matching_data_title = [df['Title'].tolist()[i] for i in sorted_indices]
    matching_data_summary = [df['Summary'].tolist()[i] for i in sorted_indices]

    top_k = k

    return matching_data_title[:top_k],matching_data_summary[:top_k]


"""**************Этим кодом один раз заполняем БД pinecone **************"""
ids = list(str(x) for x in range(len(q1)))  

def upload():
        question_list = []
        for i,row in df.iterrows():
          question_list.append(
              
                {'id' : str(i),
                'values': q1[i].tolist(),
                'metadata':{
                    "Title":row['Title'],
                    "Summary": row['Summary']
                }}
              
                                ) 

        index_name.upsert(question_list[:580]) # только столько записей туда помещаются

# upload()

def similarity_search_pinecone(inp,k):
    if len(inp) in range(2):
        return "No Results"
    que=inp
    que=model.encode(que).tolist()
    res=index_name.query(
        vector = que, top_k=k, include_metadata=True
        )
    id=0
    Title=[]
    summ=[]

    for x in res['matches']:
      Title.append(x['metadata']['Title'])
      summ.append(x['metadata']['Summary'])


    return Title,summ



def similarity_search_ANN(inp,k):
    vector_dim = len(q1[0])
    annoy_index = AnnoyIndex(vector_dim, 'angular')


    for i, vector in enumerate(q1):
        annoy_index.add_item(i, vector)
    num_trees = 100
    annoy_index.build(num_trees)
    query = inp
    query_vector = model.encode([query])[0]
    num_neighbors = k  
    nearest_neighbors = annoy_index.get_nns_by_vector(query_vector, num_neighbors)
    nearest_documents_title = [df['Title'][i] for i in nearest_neighbors]
    nearest_documents_summary = [df['Summary'][i] for i in nearest_neighbors]

    return nearest_documents_title,nearest_documents_summary




print("Всё OK")


