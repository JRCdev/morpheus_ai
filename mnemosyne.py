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
import charset_normalizer
import json

random.seed()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# text block size
tbs = 350
hash_library = {}

def pare_down(input):
  input = input.lower()
  input = re.sub(r"[\W\s]+", " ", input)
  return input

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = charset_normalizer.detect(f.read(10000))  # Read first 10KB
        return result['encoding']

def encode_with_mem(instr, model):
  # Calculate hash of the input string
  hash_value = hash(instr)
  
  # Check if hash exists in library
  if hash_value in hash_library:
      return hash_library[hash_value]
  
  # If not in library, compute f() and store result
  result = model.encode(instr).tolist()
  hash_library[hash_value] = result
  return result


def weighted_encode(strs, weights, model):
  for i, x in enumerate(strs):
    if x is None or len(x) == 0:
      strs[i] = "dummy"
      weights[i] = 0

  vectors = [encode_with_mem(x, model) for x in strs]
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

def chunk_operations(operations, chunk_size=100):
    for i in range(0, len(operations), chunk_size):
        yield operations[i:i + chunk_size]

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
        "title": f"{title} pt {str(1+x)}/{str(1+len(text)//tbs)}",
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

    title_elem = meta_tree.find('//{http://purl.org/dc/elements/1.1/}title')
    title = title_elem.text if title_elem is not None else meta.split("/")[-2]

    author_elem = meta_tree.find('//{http://purl.org/dc/elements/1.1/}creator')
    author = author_elem.text if author_elem is not None else meta.split("/")[-3]

    description_elem = meta_tree.find('//{http://purl.org/dc/elements/1.1/}description')
    description = description_elem.text if description_elem is not None else f"This is an excerpt from the book {title} by {author}"

    pddesc = pare_down(description)
    try:
      text = textract.process(book_addr, method='pdftotext', encoding='utf-8').decode('utf-8', errors='ignore').split(" ")
    except UnicodeDecodeError:
      text = textract.process(book_addr, method='tesseract', encoding='utf-8').decode('utf-8', errors='ignore').split(" ")
    operations = []
    for x in range(len(text)//tbs):
      line = re.sub(r"\s+", " ", " ".join(text[x*tbs:min(len(text),(x+1)*tbs)]).replace("\n", " "))
      #print(line)
      operations.append({"index": {"_index": "book_index"}})
      new_op = {
        "title": f"{title} pt {str(1+x)}/{str(1+len(text)//tbs)}",
        "text_vector": weighted_encode( [author, title, description, tstr, line], [5, 5, 5, 5, 80], model),
        "text":line,
        "description":description,
        "author":author
      }
      if len(tags) > 0:
        new_op["tags"] = tags
      operations.append(new_op)
    if len(operations) > 0:
      for chunk in chunk_operations(operations):
        try:
            client.bulk(index="book_index", operations=chunk, refresh=True)
        except Exception as e:
            print(f"Bulk indexing error: {e}")
            errs.append([book_addr, meta, str(e)])
    hash_library = {}
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

    text = (" ".join(open(transcript,"r").readlines()[9::8])).split(" ")
    operations = []
    for x in range(len(text)//tbs):
      line = " ".join(text[x*tbs:min(len(text),(x+1)*tbs)])
      #print(len(line))
      operations.append({"index": {"_index": "book_index"}})
      operations.append({
        "title": f"{title} pt {str(1+x)}/{str(1+len(text)//tbs)}",
        "text_vector":weighted_encode([author, title, description, line], [5, 5, 5, 80], model),
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

