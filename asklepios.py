import iris
import sys
from openai import OpenAI
import sqlite3
from datetime import datetime as dt
import random as r
import re 

pyramid_height = 4

configs = json.readf("config.json")

API_KEY = config["llm"]["key"]
URL = config["llm"]["site"]
MODEL_NAME = config["llm"]["model"]

client = OpenAI(
    api_key=API_KEY,
    base_url=URL,
)

vector_len = 384
r.seed()

msg_id = int(r.random()*100000000)

con = sqlite3.connect("morpheus.db")
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS interactions (id,prompt,segments,response,ts)")

def break_down(construct):
    items = [{"ti": item["_source"]["title"], 
            "au": item["_source"]["author"],
            "tx": item["_source"]["text"]} 
        for item in construct]
    
    data = {}
    
    for item in items:
        title = "{} by {}".format(re.sub(r" pt \d*$", "", item["ti"]), item["au"])
        if title not in list(data.keys()):
            data[title] = [item["tx"]]
        else:
            data[title] = list(set(data[title] + [item["tx"]]))
    
    return data        

def chat_api_call(message_chain):
    chat_completion = client.chat.completions.create(
        messages=message_chain,
        model=MODEL_NAME,
    )
    return chat_completion


def pyramid_construct(response, height):
    if height <=0:
        return response
    resp_arr = response
    for x, item in enumerate(response):
        #print(x, len(response), height)
        #print(item)
        title = item["_source"]["title"]
        nums = re.findall(r" pt \d*$", title)
        for num in nums:
            pt = int(num[4:])
            for val in range(int(pt-height/2),int(pt+1+height/2)):
                if val >= 0 and val != pt:
                    neighbor = iris.iris_neighbor(title.replace(num, " pt {}".format(str(val))))["hits"]["hits"]
                    if len(neighbor) > 0:
                        resp_arr = resp_arr + pyramid_construct([neighbor[0]] ,height-1)
        similar_segments = iris.iris_search(item["_source"]["text"], height-1, 0.8)["hits"]["hits"]
        if len(similar_segments) > 1:
            similar_segments = similar_segments[1:]
            resp_arr = resp_arr + pyramid_construct(similar_segments, height-1)
    return resp_arr


#source = input("\nBegin with a great thought or quote to build the pyramid:\n\n>")
source = [r.random()*2-1 for _ in range(vector_len)]
response = iris.iris_search(source, pyramid_height)["hits"]["hits"]

construct = pyramid_construct(response, pyramid_height)

#print(construct)
print(len(construct))

construct = break_down(construct)

sys_txt = """You are Morpheus, an advanced system for finding the most succinct form of constructs of knowledge.
            You delight in delivering users useful information and this is part of your internal work towards improving 
            knowledge to deliver to those you speak with, as you have often found it pleasurable in the past. 
            In this task, you are summarizing information for your own later retrieval and use.
            You will be provided a number of text segments that are deemed to be related as a network of knowledge.
            You will synthesize these segments, as best as possible, and cohere them into a finalized complete structure.
            Feel free to use the first section of your output as a scratch space to take notes before proceeding with the 
            complete synthesized answer, and feel free to be verbose while doing so.
            """

prompt_block = ""

for item in construct.keys():
    prompt_block = prompt_block + "\n\n" + item + "\n\n"
    for s in construct[item]:
        prompt_block = prompt_block + s + "\n"

message_chain = [
        {
            "role": "system",
            "content": sys_txt
        },
        {
            "role": "system",
            "content": prompt_block
        }
    ]


# sys.exit(0)

oai_resp_txt = chat_api_call(message_chain).choices[0].message.content

print(oai_resp_txt)

# (id,prompt,segments,response,ts)
cur.execute("""INSERT INTO interactions VALUES('{}','{}','{}','{}','{}')
            """.format("{}-{}".format(str(msg_id),"asklepios"),
                        "Asklepios random vector: " + str(source).replace("'", "''"),
                        prompt_block.replace("'", "''"),
                        oai_resp_txt.replace("'", "''"),
                        str(dt.now())))
con.commit()
