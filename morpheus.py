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

#console = Console()

configs = json.readf("config.json")

API_KEY = config["llm"]["key"]
URL = config["llm"]["site"]
MODEL_NAME = config["llm"]["model"]


with open("./prompt.txt", "r") as p:
    PROMPT = p.read()

r.seed()
fade = (1 + 5 ** 0.5) / 2 # golden ratio

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



# # Function to replace bold/italic with Unicode styles
# def preprocess_markdown(text):
#     # Unicode mapping for bubble letters (partial example)
#     bubble_map = {
#         'A': '🅐', 'B': '🅑', 'C': '🅒', 'D': '🅓', 'E': '🅔', 'F': '🅕', 'G': '🅖',
#         'H': '🅗', 'I': '🅘', 'J': '🅙', 'K': '🅚', 'L': '🅛', 'M': '🅜', 'N': '🅝',
#         'O': '🅞', 'P': '🅟', 'Q': '🅠', 'R': '🅡', 'S': '🅢', 'T': '🅣', 'U': '🅤',
#         'V': '🅥', 'W': '🅦', 'X': '🅧', 'Y': '🅨', 'Z': '🅩',
#         'a': '🅐', 'b': '🅑', 'c': '🅒', 'd': '🅓', 'e': '🅔', 'f': '🅕', 'g': '🅖',
#         'h': '🅗', 'i': '🅘', 'j': '🅙', 'k': '🅚', 'l': '🅛', 'm': '🅜', 'n': '🅝',
#         'o': '🅞', 'p': '🅟', 'q': '🅠', 'r': '🅡', 's': '🅢', 't': '🅣', 'u': '🅤',
#         'v': '🅥', 'w': '🅦', 'x': '🅧', 'y': '🅨', 'z': '🅩',
#         ' ': ' '
#     }

#     # Mapping for mathematical italic (for "italic" effect)
#     math_italic_map = {
#         'A': '𝐴', 'B': '𝐵', 'C': '𝐶', 'D': '𝐷', 'E': '𝐸', 'F': '𝐹', 'G': '𝐺',
#         'H': '𝐻', 'I': '𝐼', 'J': '𝐽', 'K': '𝐾', 'L': '𝐿', 'M': '𝑀', 'N': '𝑁',
#         'O': '𝑂', 'P': '𝑃', 'Q': '𝑄', 'R': '𝑅', 'S': '𝑆', 'T': '𝑇', 'U': '𝑈',
#         'V': '𝑉', 'W': '𝑊', 'X': '𝑋', 'Y': '𝑌', 'Z': '𝑍',
#         'a': '𝑎', 'b': '𝑏', 'c': '𝑐', 'd': '𝑑', 'e': '𝑒', 'f': '𝑓', 'g': '𝑔',
#         'h': 'ℎ', 'i': '𝑖', 'j': '𝑗', 'k': '𝑘', 'l': '𝑙', 'm': '𝑚', 'n': '𝑛',
#         'o': '𝑜', 'p': '𝑝', 'q': '𝑞', 'r': '𝑟', 's': '𝑠', 't': '𝑡', 'u': '𝑢',
#         'v': '𝑣', 'w': '𝑤', 'x': '𝑥', 'y': '𝑦', 'z': '𝑧',
#         ' ': ' '
#     }

#     def to_unicode_style(text, char_map):
#         return ''.join(char_map.get(c, c) for c in text)
    
#     # Replace **bold** with bubble letters
#     text = re.sub(r'\*\*(.*?)\*\*', lambda m: to_unicode_style(m.group(1), bubble_map), text)
#     # Replace *italic* with math italic
#     text = re.sub(r'\*(.*?)\*', lambda m: to_unicode_style(m.group(1), math_italic_map), text)
#     return text


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
    start_txt = "\nECHELON IV\nMORPHEUS ONLINE\nHow may I assist you?"
    print(start_txt)
    text_input = big_input("> ")
    convo = [text_input]
    print("\n")
    response = iris.iris_convo(convo, fade, 30, 0.72)
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


        
        
