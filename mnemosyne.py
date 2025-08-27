# Mnemosyne
# Goddess of Memory
# Recreates Elasticsearch from documents

import os 
import textract 
from hashlib import md5
import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import re
import ffmpeg
from llama_cpp import Llama
from tqdm import tqdm
import sqlite3
import random
import json

random.seed()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# text block size
tbs = 350

top = []

with open("./100_most_common.txt") as f:
  top = [l.strip().lower() for l in f.readlines()]

def pare_down(input):
  input = input.lower()
  input = re.sub(r"[\W\s]+", " ", input)
  for w in top:
    input = input.replace(f" {w} ", " ")
  return input

# Useful links for extracting data:

# https://textract.readthedocs.io/en/stable/installation.html
# https://github.com/abetlen/llama-cpp-python
# https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
# https://docs.python.org/3/library/xml.etree.elementtree.html
# https://stackoverflow.com/questions/51342429/how-to-extract-metadata-of-video-files-using-python-3-7
# vtt segments: x[10::8]

def ebook_match(nm):
  return (".epub" in nm or ".pdf" in nm) and nm[:2] != "._"

with open('config.json', 'r') as file:
    configs = json.load(file)

client = Elasticsearch(
    # For local development
    hosts=[configs["elasticsearch"]["host"]],
    #cloud_id=configs["elasticsearch"]["name"],
    api_key=configs["elasticsearch"]["key"],
)

print(client.info())

model = SentenceTransformer(configs["elasticsearch"]["model"])

#print("Getting book data")

# Path to the tiny tag/summarization model
model_path = configs["elasticsearch"]["model"]

# Absolute path to Calibre Library directory
book_source_directory = configs["data"]["ebooks"]
vid_source_directory = configs["data"]["transcripts"]

book_list = []
book_dir_walk = os.walk(book_source_directory,topdown=True)

vid_list =[]
vid_dir_walk = os.walk(vid_source_directory,topdown=True)

# Get a list of video files and their respective metadata files
for (root,dirs,files) in vid_dir_walk:
  vids_filtered = list(filter(lambda x: x.endswith(".en.vtt"), files))
  for vid in vids_filtered:
    vid_file = list(filter(lambda x: x.startswith(vid[:-8]) and ".vtt" not in x and ".live_chat.json" not in x, files))[0]
    vid_list.append([root + "/" + vid, root + "/" + vid_file])
#print(vid_list)

# Get a list of book files and their respective metadata files
for (root,dirs,files) in book_dir_walk:
  book_filtered = list(filter(ebook_match, files))
  #print(book_filtered)
  if len(book_filtered) > 0:
    book_list.append([root + "/" + book_filtered[0], root + "/" + "metadata.opf"])

random.shuffle(vid_list)
random.shuffle(book_list)

# Start the database and create tables if they do not exist
# Define the mapping
mappings = {
    "properties": {
        "text_vector": {
            "type": "dense_vector",
            "dims": 384,
            "index": "true",
            "similarity": "cosine",
        },
    },
}

con = sqlite3.connect("morpheus.db")
cur = con.cursor()
x = cur.execute("SELECT prompt,response,id FROM INTERACTIONS")


# Delete and recreate the index
client.indices.delete(index="book_index", ignore_unavailable=True)
client.indices.create(index="book_index", mappings=mappings)

errs = []

# Load in segments
for item in tqdm(x.fetchall()):
  try:
    title = item[0]
    author = "MORPHEUS AI"
    description = "Record of a previous interaction with MORPHEUS"
    text = item[1].split(" ")
    operations = []
    for x in range(len(text)//tbs):
      line = " ".join(text[x*tbs:min(len(text),(x+1)*tbs)])
      #print(len(line))
      operations.append({"index": {"_index": "book_index"}})
      operations.append({
        "title":"{} pt {}".format(title,str(x)),
        "text_vector":model.encode(line).tolist(),
        "text":line,
        "description":description,
        "author":author
      })

    if len(operations) > 0:
      client.bulk(index="book_index", operations=operations, refresh=True)
  except Exception as e:
    print(e)
    errs.append([item[2],str(e)])


# Get metadata across all books
for item in tqdm(book_list):
  try:
    book_addr = item[0]
    #print(book_addr)
    meta = item[1]
    meta_tree = ET.parse(item[1])
    tags = [x.text for x in meta_tree.findall('.//{http://purl.org/dc/elements/1.1/}subject')]
    tstr = " ".join(tags)
    #print(tags)
    title = meta_tree.find('.//{http://purl.org/dc/elements/1.1/}title').text
    #print(title)
    author = meta_tree.find('.//{http://purl.org/dc/elements/1.1/}creator').text
    #print(author)
    description = meta_tree.find('.//{http://purl.org/dc/elements/1.1/}description')
    if description != None:
      description = description.text
    else:
      description = "This is an excerpt from the book {} by {}".format(title, author)
    pddesc = pare_down(description)
    #print(description)
    # with open(book_addr, 'rb') as f:
    #   hash = str(md5(f.read()).hexdigest()[:10])
    #print(hash)
    text = textract.process(book_addr).decode("utf-8", errors="ignore").split(" ")
    operations = []
    for x in range(len(text)//tbs):
      line = re.sub(r"\s+", " ", " ".join(text[x*tbs:min(len(text),(x+1)*tbs)]).replace("\n", " "))
      #print(line)
      operations.append({"index": {"_index": "book_index"}})
      new_op = {
        "title":"{} pt {}".format(title,str(x)),
        "text_vector":model.encode(f"{author} {title} {pddesc} {tstr} {line}").tolist(),
        "text":line,
        "description":description,
        "author":author
      }
      if len(tags) > 0:
        new_op["tags"] = tags
      operations.append(new_op)
    if len(operations) > 0:
      client.bulk(index="book_index", operations=operations, refresh=True)
  except Exception as e:
    print(e)
    errs.append([item[0],item[1],str(e)])


# Get metadata across all videos
for item in tqdm(vid_list):
  try:
    transcript = item[0]
    vid_addr = item[1]
    #print(vid_addr)
    title = transcript.split("/")[-1].replace(".en.vtt", "")
    #print(title)
    author = vid_addr.split("/")[-3]
    #print(author)
    description = ffmpeg.probe(vid_addr)["format"]["tags"].get("DESCRIPTION", "An excerpt of the video {} by {}".format(title,author))
    pddesc = pare_down(description)
    #print(description)
    # with open(transcript, 'rb') as f:
    #   hash = str(md5(f.read()).hexdigest()[:10])
    #print(hash)
    text = (" ".join(open(transcript,"r").readlines()[9::8])).split(" ")
    operations = []
    for x in range(len(text)//tbs):
      line = " ".join(text[x*tbs:min(len(text),(x+1)*tbs)])
      #print(len(line))
      operations.append({"index": {"_index": "book_index"}})
      operations.append({
        "title":"{} pt {}".format(title,str(x)),
        "text_vector":model.encode(f"{author} {title} {pddesc} {line}").tolist(),
        "text":line,
        "description":description,
        "author":author
      })

    if len(operations) > 0:
      client.bulk(index="book_index", operations=operations, refresh=True)
  except Exception as e:
    print(e)
    errs.append([item[0],item[1],str(e)])

print("error'd books/transcripts")
for item in errs:
  for field in item:
    print(field)
  print("\n\n")

