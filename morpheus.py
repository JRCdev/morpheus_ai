import iris
from openai import OpenAI
import sqlite3
from datetime import datetime as dt
import random as r
import math
from rich import print as xp
from rich.markdown import Markdown
#from rich.console import Console
import re
import json
import argparse
import pandas as pd 

#console = Console()

r.seed()
fade = (1 + 5 ** 0.5) / 2 # golden ratio

con = sqlite3.connect("morpheus.db")
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS interactions (id,prompt,segments,response,ts)")

# function that performs API calls
def chat_api_call(message_chain):
    chat_completion = client.chat.completions.create(
        messages=message_chain,
        model=MODEL_NAME,
    )
    return chat_completion

def big_input(first):
    """allows for multiline inputs"""
    full = ""
    prev = "x"
    while True:
        line = input(first)
        full = full + "\r\n" + line
        if len(line) == 0 and len(prev) == 0:
            return full.strip()
        prev = line


#Main function that takes in user input, gets Iris response,
#  and initiates dialogue with LLM with engineered prompt

def gen_prompt_spike(response, prompt):
    if len(response["hits"]["hits"]) == 0:
        return prompt.replace("%SEGMENTS%", "")
    else:
        replace_txt = "## Text Segments\n"
        for hit in response["hits"]["hits"]:
            id = hit["_id"]
            #publication_date = hit["_source"]["publish_date"]
            #rank = hit["_rank"]
            title = hit["_source"]["title"]
            search_title = title.replace(" ", "+").lower()
            author = hit["_source"]["author"]
            text = hit["_source"]["text"]
            pretty_output = f"""\n
            > {text}
            [{title} by {author}](https://media.johncre.ws/web/#/search.html?query={search_title})
            """
            replace_txt = replace_txt + pretty_output
        return prompt.replace("%SEGMENTS%", replace_txt)
    

# main function invocation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional Flags for the Morpheus script") 
    parser.add_argument('--search-results', type=int, default=50, help='Number of search results (default: 50)') 
    parser.add_argument('--search-temp', type=float, default=0.72, help='Threshold of Elasticsearch Documents (default: 0.72)') 
    parser.add_argument('--config', type=str, default='config.json', help='Config file path (default: config.json)') 
    parser.add_argument('--prompt', type=str, default='prompt.txt', help='Prompt file path (default: prompt.txt)')
    parser.add_argument('--convo-id', type=int, default=None, help='Conversation ID to continue from a previous chat')
    

    with open('config.json', 'r') as file:
        configs = json.load(file)

    API_KEY = configs["llm"]["key"]
    URL = configs["llm"]["site"]
    MODEL_NAME = configs["llm"]["model"]

    client = OpenAI(
        api_key=API_KEY,
        base_url=URL,
    )

    args = parser.parse_args()

    with open(args.prompt, "r") as p:
        PROMPT = p.read()

    start_txt = "\nECHELON IV\nMORPHEUS ONLINE\nHow may I assist you?"

    convo = []
    resps = [start_txt.strip()]

    if args.convo_id is not None:
        df_history = pd.read_sql_query(f"SELECT id, prompt, response, ts FROM interactions where id like '{args.convo_id}-%' order by ts", con)
        msg_id = args.convo_id
        convo = df_history['prompt'].to_list()
        resps = resps + df_history['response'].to_list()
    else:       
        msg_id = int(r.random()*100000000)

    message_no = len(convo)

    for x in range(min(len(convo),len(resps))):
        xp(Markdown(f"{resps[x]}\n+> {convo[x]}\n\n"))
    xp(Markdown(f"{resps[-1]}\n" if len(resps) > 1 else ""))

    text_input = big_input("> ")
    print("\n")

    convo.append(text_input)
    response = iris.iris_convo(resps + convo, fade, args.search_results, args.search_temp)

    message_chain = [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "tool",
                "content": gen_prompt_spike(response, "%SEGMENTS%")
            }
        ]
    
    for x in range(min(len(convo),len(resps))):
        message_chain.append({
                "role": "assistant",
                "content": resps[x]
            })
        message_chain.append({
                "role": "user",
                "content": convo[x]
            })


    while (True):
        oai_resp_txt = chat_api_call(message_chain).choices[0].message.content
        xp(Markdown(oai_resp_txt, style="markdown"))
        resps.append(oai_resp_txt)
        print("\nSources for further reading:")
        segments = "\n".join(["[{title} by {author}]".format(title=item["_source"]["title"],
                                        author=item["_source"]["author"]) for item in response["hits"]["hits"]])
        print(segments)
        cur.execute("""INSERT INTO interactions VALUES('{}','{}','{}','{}','{}')
                    """.format("{}-{}".format(str(msg_id),str(message_no)),
                               text_input.replace("'", "''"),
                               gen_prompt_spike(response,"%SEGMENTS%").replace("'", "''"),
                               oai_resp_txt.replace("'", "''"),
                               str(dt.now())))
        con.commit()
        message_no = message_no + 1

        print("")
        text_input = big_input("> ")
        convo.append(text_input)
        print("\n")
        response = iris.iris_convo(resps + convo, fade, 22 + (3 * message_no), 0.72)
        message_chain.append({
            "role": "assistant",
            "content": oai_resp_txt
        })
        message_chain[1] = {
            "role": "tool",
            "content": gen_prompt_spike(response,"%SEGMENTS%")
        }
        message_chain.append({
            "role": "user",
            "content": text_input
        })


        
        
