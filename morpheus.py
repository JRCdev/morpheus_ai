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
    parser.add_argument('--search-results', type=int, default=50, help='Number of search results (default: 25)') 
    parser.add_argument('--search-temp', type=float, default=50, help='Threshold of Elasticsearch Documents (default: 0.72)') 
    parser.add_argument('--config', type=str, default='config.json', help='Config file path (default: config.json)') 
    parser.add_argument('--prompt', type=str, default='prompt.txt', help='Prompt file path (default: prompt.txt)')
    parser.add_argument('--convo-id', type=int, default=r.random()*100000000, help='Conversation ID to continue from a previous chat')
    

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
    print(start_txt)
    text_input = big_input("> ")
    convo = [text_input]
    print("\n")
    response = iris.iris_convo(convo, fade, args.search_results, args.search_temp)
    msg_id = int(r.random()*100000000)
    message_no = 0
    message_chain = [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "tool",
                "content": gen_prompt_spike(response, "%SEGMENTS%")
            },
            {
                "role": "assistant",
                "content": start_txt.strip()
            },
            {
                "role": "user",
                "content": text_input,
            }
        ]
    while (True):
        oai_resp_txt = chat_api_call(message_chain).choices[0].message.content
        xp(Markdown(oai_resp_txt, style="markdown"))
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
        response = iris.iris_convo(convo, fade, 22 + (3 * message_no), 0.72)
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


        
        
