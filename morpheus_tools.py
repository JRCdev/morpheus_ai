import iris
from openai import OpenAI
import sqlite3
from datetime import datetime as dt
import random as r
import json

INCENTIVE = """\nIn your answer, please query the provided search function using conventional and lateral thinking, then synthesize"""

configs = json.readf("config.json")

API_KEY = config["llm"]["key"]
URL = config["llm"]["site"]
MODEL_NAME = config["llm"]["model"]

with open("./prompt.txt", "r") as p:
    PROMPT = p.read()

r.seed()

con = sqlite3.connect("morpheus.db")
cur = con.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS interactions (id,prompt,segments,response,ts)")

client = OpenAI(
    api_key=API_KEY,
    base_url=URL,
)

# function that performs API calls
def chat_api_call(message_chain):
    chat_completion = client.chat.completions.create(
        messages=message_chain,
        model=MODEL_NAME,
        tools=tools_definition,
        tool_choice="auto"
    )
    return chat_completion

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

def search_book_segments(query, num_of_articles):
    return gen_prompt_spike(iris.iris_search(text_input, num_of_articles, 0.72) , "%SEGMENTS%")

def purge_tool_messages(messages, id=None):
    new_messages = []
    for item in messages:
        if item["role"].lower() != "tool":
            new_messages.append(item)
        elif id is not None and item["tool_call_id"] != id:
            new_messages.append(item)
    return new_messages


tools_definition = [
    {
        "type": "function",
        "function": {
            "name": "search_databanks",
            "description": """Search the document library for text matches. 
                                Multiple invocations with varied prompts 
                                may be necessary for best results.
                                Inputs are vectorized and compared with the vectors of index items
                                Search engine is powered by Elasticsearch""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "str",
                        "description": "Search text to find matches for. Queries should be specific and broken up to encapsulate separate ideas."
                    },
                    "num_of_results": {
                        "type": "int",
                        "description": "Number of book pages to return. Users report best results when asking for between five and thirty items."
                    }
                },
                "required": ["location"]
            }
        }
    }
]

tools_map = {
    "search_databanks": iris.iris_search
}

# main function invocation
if __name__ == "__main__":
    start_txt = "\nECHELON IV\nMORPHEUS ONLINE\nHow may I assist you?"
    print(start_txt)
    text_input = input("\n> ")
    print("\n")
    msg_id = int(r.random()*100000000)
    message_no = 0
    message_chain = [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "assistant",
                "content": start_txt.strip()
            },
            {
                "role": "user",
                "content": text_input + INCENTIVE,
            }
        ]
    while (True):
        oai_resp = chat_api_call(message_chain).choices[0].message
        full_cite = []
        items = []
        segments = ""
        lookups = {}
        while(oai_resp.tool_calls):
            #print("tool call")
            for tool_call in oai_resp.tool_calls:
                # Get the tool function name and arguments Grok wants to call
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print("\n{}\n".format(str(function_args)))

                # AI likes to repeat search; instead replace old search with expanded results
                if function_name == "search_databanks":
                    if function_args["query"].lower() in list(lookups.keys()):
                        # Remove old call
                        message_chain = purge_tool_messages(message_chain, lookups[function_args["query"].lower()]["id"])
                        # Make new call
                        result = iris.iris_search(function_args["query"], lookups[function_args["query"].lower()]["ct"]+function_args["num_of_results"])
                        # Update dictionary
                        lookups[function_args["query"].lower()]["ct"] = lookups[function_args["query"].lower()]["ct"]+function_args["num_of_results"]
                    else:
                        lookups[function_args["query"].lower()] = {"id": tool_call.id, "ct":function_args["num_of_results"]}
                        result = iris.iris_search(function_args["query"], function_args["num_of_results"])
                else:
                    result = tools_map[function_name](**function_args)

                result_txt = gen_prompt_spike(result , "%SEGMENTS%")
                full_cite.append(result_txt)
                segments += "\n".join(["[{title} by {author}]".format(title=item["_source"]["title"],
                                        author=item["_source"]["author"]) for item in result["hits"]["hits"]]) + "\n"
                
                # Append the result from tool function call to the chat message history,
                # with "role": "tool"
                message_chain.append(
                    {
                        "role": "tool",
                        "content": result_txt,
                        "tool_call_id": tool_call.id  # tool_call.id supplied in Grok's response
                    }
                )
            oai_resp = chat_api_call(message_chain).choices[0].message
        
        oai_resp_txt = oai_resp.content
        print(oai_resp_txt)
        if len(segments) > 0:
            print("\nSources for further reading:")
            print(segments)
        cur.execute("""INSERT INTO interactions VALUES('{}','{}','{}','{}','{}')
                    """.format("{}-{}".format(str(msg_id),str(message_no)),
                               text_input.replace("'", "''"),
                               "\n".join(full_cite).replace("'", "''"),
                               oai_resp_txt.replace("'", "''"),
                               str(dt.now())))
        con.commit()
        message_no = message_no + 1

        text_input = input("\n> ")
        print("")
        message_chain = purge_tool_messages(message_chain)
        message_chain.append({
            "role": "assistant",
            "content": oai_resp_txt
        })
        message_chain.append({
            "role": "user",
            "content": text_input + INCENTIVE
        })


        
        