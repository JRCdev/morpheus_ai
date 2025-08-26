# Iris
# A messenger goddess
# Takes in question, returns most relevant segments

import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def pretty_print_response(response):
    if len(response["hits"]["hits"]) == 0:
        print("Your search returned no results.")
    else:
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            #publication_date = hit["_source"]["publish_date"]
            #rank = hit["_rank"]
            title = hit["_source"]["title"]
            author = hit["_source"]["author"]
            text = hit["_source"]["text"]
            description = hit["_source"]["description"]
            tags = hit["_source"].get("tags", [])
            pretty_output = f"""\nID: {id}
            Title: {title}
            Author: {author}
            Summary: {description}
            Tags: {tags}
            Text: {text}"""
            print(pretty_output)

configs = json.readf("config.json")

client = Elasticsearch(
    # For local development
    hosts=[configs["elasticsearch"]["host"]],
    #cloud_id=configs["elasticsearch"]["name"],
    api_key=configs["elasticsearch"]["key"],
)

print(client.info())

def weighted_encode(strs, weights, model):
  for i, x in enumerate(strs):
    if x is None or len(x) == 0:
      strs[i] = "dummy"
      weights[i] = 0

  vectors = [model.encode(x).tolist() for x in strs]
  vector_len = len(vectors[0])
  weight_sum = sum(weights)

  if weight_sum == 0:
      raise ValueError("Sum of weights cannot be zero")
  
  result = [0] * vector_len
  
  for i in range(vector_len):
      for vec, weight in zip(vectors, weights):
          result[i] += vec[i] * weight
  
  # Divide by sum of weights to get weighted average
  return [x / weight_sum for x in result]

def iris_convo(queries, fade, num_of_results=5, relevance = 0.5):
    weights = []
    for i, x in enumerate(queries):
        if i == 0:
            weights.append(1)
        else:
            weights.append(weights[-1] / fade)
    vector = weighted_encode(strs=queries, weights=weights, model=model)
    return iris_search(vector,num_of_results, relevance)

def iris_search(query, num_of_results=5, relevance = 0.5):
    #x = model.encode(query.lower()).tolist()
    #print(str(type(x)))
    response = client.search(
        index="book_index",
        size=num_of_results,
        query={"match": {"summary": str(query)}},
        knn={
            "field": "text_vector",
            "query_vector": query if isinstance(query, list) else model.encode(query.lower()).tolist(),  
            "k": num_of_results+2,
            "num_candidates": (num_of_results+2) * 100,
        },
        min_score = relevance
    )
    return response

def iris_neighbor(query):
    response = client.search(
        index="book_index",
        size=3,
        body={"query": {"match": {"title": query}}},
    )
    return response


if __name__ == "__main__":
    while (True):
        response = iris_search(input("\nEnter text query\n\n> "), 10, 0.656565656565)
        pretty_print_response(response)
