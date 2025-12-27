# Mnemosyne
# Goddess of Memory
# Recreates Elasticsearch from documents

import os 
import textract 
import hashlib
import time
# ProcessPoolExecutor was previously considered but not used; avoid accidental pools
import xml.etree.ElementTree as ET
from elasticsearch import Elasticsearch, exceptions as es_exceptions
from sentence_transformers import SentenceTransformer
import re
import ffmpeg
from llama_cpp import Llama
from tqdm import tqdm
import sqlite3
import random
import charset_normalizer
import json
from pathlib import Path
from kerukeion import n_rare_words
import argparse
import numpy as np
import gc

random.seed()

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# text block size
tbs = 350
# persistent in-sqlite cache will be used instead of the old in-memory hash_library

# optional third-party utilities
try:
  import webvtt
except Exception:
  webvtt = None

try:
  import easyocr
  import pdf2image
except Exception:
  easyocr = None
  pdf2image = None

_easyocr_readers = {}


def _iter_pdf_pages_in_chunks(pdf_path, dpi=200, chunk_size=1):
  """Yield PIL pages from a PDF in small chunks to limit peak memory.

  This uses pdf2image.pdfinfo_from_path to determine number of pages and
  calls convert_from_path with first_page/last_page so we never hold the
  entire PDF as a large list of images.
  """
  if pdf2image is None:
    raise RuntimeError("pdf2image is not available")
  try:
    info = pdf2image.pdfinfo_from_path(pdf_path)
    # pdfinfo keys may vary; prefer 'Pages' then 'pages'
    num_pages = int(info.get('Pages') or info.get('pages') or 0)
  except Exception:
    # fallback: try convert_from_path and yield as-is (best-effort)
    pages = pdf2image.convert_from_path(pdf_path, dpi=dpi)
    for p in pages:
      yield p
    return

  if num_pages <= 0:
    # fallback single pass
    pages = pdf2image.convert_from_path(pdf_path, dpi=dpi)
    for p in pages:
      yield p
    return

  for start in range(1, num_pages + 1, max(1, chunk_size)):
    end = min(num_pages, start + chunk_size - 1)
    pages = pdf2image.convert_from_path(pdf_path, dpi=dpi, first_page=start, last_page=end)
    for p in pages:
      yield p
    # release memory for this chunk promptly
    del pages
    gc.collect()

def pare_down(input):
  input = input.lower()
  input = re.sub(r"[\W\s]+", " ", input)
  return input

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = charset_normalizer.detect(f.read(10000))  # Read first 10KB
        return result['encoding']

def md5_hash(text: str) -> str:
  return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_cached_vector(cur, text_hash: str):
  # Respect runtime flag to disable cache
  if 'USE_CACHE' in globals() and not USE_CACHE:
    return None
  row = cur.execute("SELECT vector_json FROM vector_cache WHERE item_hash=?", (text_hash,)).fetchone()
  return None if row is None else json.loads(row[0])


def set_cached_vector(con, cur, text_hash: str, vector):
  # Respect runtime flag to disable cache writes
  if 'USE_CACHE' in globals() and not USE_CACHE:
    return
  cur.execute("INSERT OR REPLACE INTO vector_cache(item_hash, vector_json, updated_at) VALUES (?, ?, datetime('now'))",
              (text_hash, json.dumps(vector)))
  con.commit()


def encode_texts_with_cache(texts, model, cur, con, batch_size=64):
  hashes = [md5_hash(t) for t in texts]
  cached = {}
  to_encode = []
  to_encode_idx = []
  for i, h in enumerate(hashes):
    v = get_cached_vector(cur, h)
    if v is not None:
      cached[i] = v
    else:
      to_encode.append(texts[i])
      to_encode_idx.append(i)

  if len(to_encode) > 0:
    encoded = model.encode(to_encode, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=False)
    for idx, vec in zip(to_encode_idx, encoded):
      vec_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
      set_cached_vector(con, cur, hashes[idx], vec_list)
      cached[idx] = vec_list

  result = [cached[i] for i in range(len(texts))]
  return result


def weighted_encode(strs, weights, model, cur, con, batch_size=64):
  for i, x in enumerate(strs):
    if x is None or len(x) == 0:
      strs[i] = "dummy"
      weights[i] = 0

  vectors = encode_texts_with_cache(strs, model, cur, con, batch_size=batch_size)
  vector_len = len(vectors[0])
  weight_sum = sum(weights)

  if weight_sum == 0:
    raise ValueError("Sum of weights cannot be zero")

  result = [0.0] * vector_len
  for vec, weight in zip(vectors, weights):
    for i in range(vector_len):
      result[i] += vec[i] * weight

  return [x / weight_sum for x in result]

def chunk_operations(operations, chunk_size=500):
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
  # api_key=configs["elasticsearch"]["key"],
  # sensible defaults to tolerate slow networks and transient ES hiccups
  request_timeout=60,
  max_retries=5,
  retry_on_timeout=True,
)

print(client.info())

# Path to the tiny tag/summarization model
model_path = configs["elasticsearch"].get("model", "all-MiniLM-L6-v2")
model_dims = configs["elasticsearch"].get("vectors_dim", 384)
model = SentenceTransformer(model_path)

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


# Start the database and create tables if they do not exist
# Define the mapping
mappings = {
    "properties": {
        "text_vector": {
            "type": "dense_vector",
            "dims": 1024,
            "index": "true",
            "similarity": "cosine",
        },
    },
}

con = sqlite3.connect("morpheus.db")
cur = con.cursor()

# create helper tables for incremental indexing and vector cache
cur.execute("""
CREATE TABLE IF NOT EXISTS indexed_files (
  path TEXT PRIMARY KEY,
  mtime INTEGER,
  content_hash TEXT,
  indexed_at TEXT,
  doc_count INTEGER
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS vector_cache (
  item_hash TEXT PRIMARY KEY,
  vector_json TEXT,
  updated_at TEXT
)
""")
con.commit()

# Parse CLI flags to control caching/reindex behavior
parser = argparse.ArgumentParser(description='Mnemosyne indexer')
parser.add_argument('--no-cache', action='store_true', help='Do not use persistent vector cache (recompute embeddings)')
parser.add_argument('--force-reindex', action='store_true', help='Ignore indexed_files and reprocess all source files')
parser.add_argument('--clear-cache', action='store_true', help='Clear the vector_cache table before starting')
parser.add_argument('--atomic-reindex', action='store_true', help='Build into a temporary index and atomically swap an alias to it when complete')
parser.add_argument('--drop-old-index', action='store_true', help='When used with --atomic-reindex, delete old physical indices that the alias pointed to')
parser.add_argument('--vacuum-cache', action='store_true', help='Run VACUUM on the sqlite DB to reclaim space (use after --clear-cache)')
parser.add_argument('--pdf-page-chunk', type=int, default=1, help='Number of PDF pages to convert in memory at once (lower reduces memory)')
args = parser.parse_args()

USE_CACHE = not args.no_cache
FORCE_REINDEX = args.force_reindex
CLEAR_CACHE = args.clear_cache
ATOMIC_REINDEX = args.atomic_reindex
DROP_OLD_INDEX = args.drop_old_index
VACUUM_CACHE = args.vacuum_cache
PDF_PAGE_CHUNK = max(1, int(args.pdf_page_chunk))

if CLEAR_CACHE:
  try:
    cur.execute("DELETE FROM vector_cache")
    con.commit()
    print('Cleared vector_cache table')
  except Exception as e:
    print('Failed to clear vector_cache:', e)

# Optionally VACUUM the sqlite DB to reclaim space after clearing cache
if VACUUM_CACHE:
  try:
    print('Running VACUUM on morpheus.db to reclaim space...')
    cur.execute('VACUUM')
    con.commit()
    print('VACUUM completed')
  except Exception as e:
    print('VACUUM failed:', e)

x = cur.execute("SELECT prompt,response,id FROM INTERACTIONS")

# Logical alias used by the rest of the system
LOGICAL_INDEX = "book_index"

# If atomic reindex requested, create a temporary physical index and write to that.
# After indexing completes we'll swap the alias atomically.
if ATOMIC_REINDEX:
  temp_index = f"{LOGICAL_INDEX}_reindex_{int(time.time())}"
  try:
    # create the temp index with same mappings
    client.indices.create(index=temp_index, mappings=mappings)
    print(f"Created temporary index {temp_index} for atomic reindex")
  except Exception as e:
    print(f"Warning creating temporary index {temp_index}: {e}")
  TARGET_INDEX = temp_index
else:
  TARGET_INDEX = LOGICAL_INDEX
  # Ensure the logical index exists (non-atomic path)
  try:
    if not client.indices.exists(index=LOGICAL_INDEX):
      client.indices.create(index=LOGICAL_INDEX, mappings=mappings)
      print(f"Created index {LOGICAL_INDEX}")
  except Exception as e:
    print(f"Warning creating index: {e}")

errs = []

# Quick health check of the ES cluster to provide an early, actionable warning
try:
  # ping is lightweight; cluster.health gives more detail and can wait
  if not client.ping():
    print('Warning: Elasticsearch ping failed. The cluster may be unreachable.')
  else:
    try:
      health = client.cluster.health(request_timeout=10)
      status = health.get('status')
      print(f"Elasticsearch cluster status: {status}")
    except Exception:
      # non-fatal; we already know ping succeeded
      pass
except Exception as e:
  print(f"Warning checking Elasticsearch cluster health: {e}")


def es_bulk_with_retries(client, operations, index, max_attempts=5, base_timeout=60):
  """Perform a bulk request with simple retry/backoff on timeout or connection errors.

  operations: the list of operations (a chunk) to send to bulk
  index: physical index name
  """
  attempt = 0
  while attempt < max_attempts:
    try:
      # Use the options() helper to set per-call transport options (avoids deprecation warnings)
      es_call = client.options(request_timeout=base_timeout)
      res = es_call.bulk(index=index, operations=operations)
      # If there are errors in the bulk response, return it so caller can inspect
      if isinstance(res, dict) and res.get('errors'):
        # log top-level error and return for caller to handle
        print(f"Bulk completed with partial errors on attempt {attempt+1}")
      return res
    except (es_exceptions.ConnectionTimeout, es_exceptions.TransportError, es_exceptions.ConnectionError) as e:
      attempt += 1
      # exponential backoff with cap and a little jitter
      wait = min(base_timeout * (2 ** (attempt-1)), 120)
      jitter = min(5, int(wait * 0.1))
      sleep_for = wait + (random.random() * jitter)
      print(f"Bulk attempt {attempt} failed for index {index}: {e}. Retrying in {int(sleep_for)}s...")
      time.sleep(sleep_for)
  # final attempt (raise last exception)
  try:
    es_call = client.options(request_timeout=base_timeout)
    return es_call.bulk(index=index, operations=operations)
  except Exception as e:
    print(f"Final bulk attempt failed for index {index}: {e}")
    raise
items_interactions = x.fetchall()
pbar_inter = tqdm(items_interactions, desc="interactions")
for item in pbar_inter:
  try:
    title = item[0]
    short_title = n_rare_words(title, 10)
    author = "MORPHEUS AI"
    # show a concise preview in the progress bar
    try:
      pbar_inter.set_description(f"interaction: {author} - {title[:40]}")
    except Exception:
      pass
    description = "Record of a previous interaction"
    full_text = item[1]
    interaction_id = item[2]

    source_key = f"interaction:{interaction_id}"
    content_hash = md5_hash(full_text)
    if not FORCE_REINDEX:
      row = cur.execute("SELECT content_hash FROM indexed_files WHERE path=?", (source_key,)).fetchone()
      if row is not None and row[0] == content_hash:
        # already indexed and unchanged
        continue

    words = full_text.split(" ")
    lines = [" ".join(words[i*tbs:min(len(words),(i+1)*tbs)]) for i in range(max(1, len(words)//tbs))]
    # ensure at least one chunk
    if len(lines) == 0 and len(full_text.strip()) > 0:
      lines = [full_text.strip()]

    vectors = encode_texts_with_cache(lines, model, cur, con)

    operations = []
    for idx, (line, vec) in enumerate(zip(lines, vectors)):
      doc_id = md5_hash(f"{source_key}|{idx}|{line[:256]}")
      operations.append({"create": {"_index": TARGET_INDEX, "_id": doc_id}})
      operations.append({
        "title": f"{short_title} pt {str(1+idx)}/{str(len(lines))}",
        "text_vector": vec,
        "text": line,
        "description": description,
        "author": author
      })

    if len(operations) > 0:
      for chunk in chunk_operations(operations):
        try:
          res = es_bulk_with_retries(client, chunk, TARGET_INDEX)
          # inspect response for partial errors and log
          if isinstance(res, dict) and res.get('errors'):
            errs.append([interaction_id, 'partial_errors_in_bulk'])
        except Exception as e:
          print(f"Bulk indexing error (interaction {interaction_id}): {e}")
          errs.append([interaction_id, str(e)])

    # update indexed_files entry
    cur.execute("INSERT OR REPLACE INTO indexed_files(path, mtime, content_hash, indexed_at, doc_count) VALUES (?, ?, ?, datetime('now'), ?)",
                (source_key, int(time.time()), content_hash, len(lines)))
    con.commit()
  except Exception as e:
    print(e)
    errs.append([item[2],str(e)])


# Get metadata across all books (incremental)
pbar_books = tqdm(book_list, desc="books")
for item in pbar_books:
  try:
    book_addr = item[0]
    meta = item[1]
    meta_tree = ET.parse(item[1])
    tags = [x.text for x in meta_tree.findall('.//{http://purl.org/dc/elements/1.1/}subject')]
    tstr = " ".join(tags)

    title_elem = meta_tree.find('.//{http://purl.org/dc/elements/1.1/}title')
    title = title_elem.text if title_elem is not None else meta.split("/")[-2]

    author_elem = meta_tree.find('.//{http://purl.org/dc/elements/1.1/}creator')
    author = author_elem.text if author_elem is not None else meta.split("/")[-3]

    try:
      pbar_books.set_description(f"book: {author} - {title[:40]}")
    except Exception:
      pass

    description_elem = meta_tree.find('.//{http://purl.org/dc/elements/1.1/}description')
    description = description_elem.text if description_elem is not None else f"This is an excerpt from the book {title} by {author}"

    # detect language if present in metadata (two-letter preferred)
    language_elem = meta_tree.find('.//{http://purl.org/dc/elements/1.1/}language')
    book_lang = (language_elem.text[:2].lower() if language_elem is not None and language_elem.text else 'en')

    # compute a lightweight content hash based on file stat (mtime + size)
    try:
      st = Path(book_addr).stat()
      content_hash = md5_hash(f"{st.st_mtime_ns}-{st.st_size}")
    except Exception:
      content_hash = md5_hash(book_addr)

    if not FORCE_REINDEX:
      row = cur.execute("SELECT content_hash FROM indexed_files WHERE path=?", (book_addr,)).fetchone()
      if row is not None and row[0] == content_hash:
        # unchanged
        continue

    # extract text (keep old textract fallback for many formats)
    try:
      ext = Path(book_addr).suffix.lower()
      text_words = []
      # EPUB: prefer direct extraction
      if ext == ".epub":
        try:
          raw = textract.process(book_addr, encoding='utf-8')
          if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
            text_words = raw.split()
        except Exception as e:
          print(f"EPUB extract failed {book_addr}: {e}")
          errs.append([book_addr, meta, str(e)])
          continue

      # PDF: prefer pdftotext for English PDFs; use EasyOCR+pdf2image for non-English or scanned PDFs
      elif ext == ".pdf":
        tried = False
        is_english = True
        try:
          is_english = (book_lang is None) or str(book_lang).lower().startswith('en')
        except Exception:
          is_english = True

        # Helper to initialize or reuse an easyocr reader for a lang_key
        def _get_easyocr_reader(lang_key):
          r = _easyocr_readers.get(lang_key)
          if r is None:
            try:
              r = easyocr.Reader([lang_key], gpu=False)
            except Exception:
              r = easyocr.Reader(['en'], gpu=False)
            _easyocr_readers[lang_key] = r
          return r

        # If we believe this is an English PDF, try pdftotext first (fast, no model downloads)
        if is_english:
          try:
            raw = textract.process(book_addr, method='pdftotext', encoding='utf-8')
            if isinstance(raw, bytes):
              raw = raw.decode('utf-8', errors='ignore')
            text_words = raw.split()
            tried = True
          except Exception:
            tried = False

          if not tried:
            # try tesseract fallback
            try:
              raw = textract.process(book_addr, method='tesseract', encoding='utf-8')
              if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='ignore')
              text_words = raw.split()
              tried = True
            except Exception:
              tried = False

          # If still not extracted and easyocr is available, try image OCR as a last resort
          if not tried and easyocr is not None and pdf2image is not None:
            try:
              lang = (book_lang if 'book_lang' in locals() and book_lang else 'en')[:2]
              reader = _get_easyocr_reader(lang)
              texts = []
              for pg in _iter_pdf_pages_in_chunks(book_addr, dpi=200, chunk_size=PDF_PAGE_CHUNK):
                try:
                  arr = np.array(pg)
                  res = reader.readtext(arr, detail=0)
                  if res:
                    texts.append(" ".join(res))
                finally:
                  # free page memory immediately
                  try:
                    del arr
                  except Exception:
                    pass
                  try:
                    del pg
                  except Exception:
                    pass
                  gc.collect()
              full_text = " ".join(texts)
              text_words = full_text.split()
              tried = True
            except Exception as e:
              print(f"easyocr/pdf2image failed for {book_addr}: {e}")
              tried = False

        else:
          # Non-English: prefer easyocr/pdf2image first if available (better multilingual support)
          if easyocr is not None and pdf2image is not None:
            try:
              lang = (book_lang if 'book_lang' in locals() and book_lang else 'en')[:2]
              reader = _get_easyocr_reader(lang)
              texts = []
              for pg in _iter_pdf_pages_in_chunks(book_addr, dpi=200, chunk_size=PDF_PAGE_CHUNK):
                try:
                  arr = np.array(pg)
                  res = reader.readtext(arr, detail=0)
                  if res:
                    texts.append(" ".join(res))
                finally:
                  try:
                    del arr
                  except Exception:
                    pass
                  try:
                    del pg
                  except Exception:
                    pass
                  gc.collect()
              full_text = " ".join(texts)
              text_words = full_text.split()
              tried = True
            except Exception as e:
              print(f"easyocr/pdf2image failed for {book_addr}: {e}")
              tried = False

          # If easyocr not available or failed, fall back to pdftotext/tesseract
          if not tried:
            try:
              raw = textract.process(book_addr, method='pdftotext', encoding='utf-8')
              if isinstance(raw, bytes):
                raw = raw.decode('utf-8', errors='ignore')
              text_words = raw.split()
              tried = True
            except Exception:
              try:
                raw = textract.process(book_addr, method='tesseract', encoding='utf-8')
                if isinstance(raw, bytes):
                  raw = raw.decode('utf-8', errors='ignore')
                text_words = raw.split()
                tried = True
              except Exception as ee:
                print(f"Failed to extract {book_addr}: {ee}")
                errs.append([book_addr, meta, str(ee)])
                continue

      # Other formats: let textract choose a method
      else:
        try:
          raw = textract.process(book_addr, encoding='utf-8')
          if isinstance(raw, bytes):
            raw = raw.decode('utf-8', errors='ignore')
            text_words = raw.split()
        except Exception as e:
          print(f"Failed to extract {book_addr}: {e}")
          errs.append([book_addr, meta, str(e)])
          continue
      
    except Exception as e:
      print(f"Extraction error for {book_addr}: {e}")
      errs.append([book_addr, meta, str(e)])
      continue

    # prepare static vectors (author/title/description/tags)
    static_texts = [author or "", title or "", description or "", tstr or ""]
    static_vecs = encode_texts_with_cache(static_texts, model, cur, con)

    # chunk the words
    lines = [re.sub(r"\s+", " ", " ".join(text_words[i*tbs:min(len(text_words),(i+1)*tbs)]).replace("\n", " ")) for i in range(max(1, len(text_words)//tbs))]
    if len(lines) == 0 and len(text_words) > 0:
      lines = [" ".join(text_words)]

    operations = []
    for idx, line in enumerate(lines):
      # encode line (with cache)
      line_vec = encode_texts_with_cache([line], model, cur, con)[0]
      # weighted combine: weights [5,5,5,5,80]
      weights = [5,5,5,5,80]
      # compute weighted average using static_vecs and line_vec
      combined = []
      for i in range(len(line_vec)):
        combined.append((static_vecs[0][i]*5 + static_vecs[1][i]*5 + static_vecs[2][i]*5 + static_vecs[3][i]*5 + line_vec[i]*80) / sum(weights))

      doc_id = md5_hash(f"{book_addr}|{idx}|{line[:256]}")
      operations.append({"create": {"_index": TARGET_INDEX, "_id": doc_id}})
      new_op = {
        "title": f"{title} pt {str(1+idx)}/{str(len(lines))}",
        "text_vector": combined,
        "text": line,
        "description": description,
        "author": author
      }
      if len(tags) > 0:
        new_op["tags"] = tags
      operations.append(new_op)

    if len(operations) > 0:
      for chunk in chunk_operations(operations):
        try:
          res = es_bulk_with_retries(client, chunk, TARGET_INDEX)
          if isinstance(res, dict) and res.get('errors'):
            errs.append([book_addr, meta, 'partial_errors_in_bulk'])
        except Exception as e:
          print(f"Bulk indexing error: {e}")
          errs.append([book_addr, meta, str(e)])

    # write indexed_files entry
    try:
      mtime = int(Path(book_addr).stat().st_mtime)
    except Exception:
      mtime = int(time.time())
    cur.execute("INSERT OR REPLACE INTO indexed_files(path, mtime, content_hash, indexed_at, doc_count) VALUES (?, ?, ?, datetime('now'), ?)",
                (book_addr, mtime, content_hash, len(lines)))
    con.commit()
  except Exception as e:
    print(e)
    errs.append([item[0],item[1],str(e)])


# Get metadata across all videos (incremental)
pbar_vids = tqdm(vid_list, desc="videos")
for item in pbar_vids:
  try:
    transcript = item[0]
    vid_addr = item[1]
    title = transcript.split("/")[-1].replace(".en.vtt", "")
    author = vid_addr.split("/")[-3]
    try:
      pbar_vids.set_description(f"video: {author} - {title[:40]}")
    except Exception:
      pass
    description = ffmpeg.probe(vid_addr)["format"]["tags"].get("DESCRIPTION", f"An excerpt of the video {title} by {author}")

    # skip unchanged transcripts
    try:
      st = Path(transcript).stat()
      content_hash = md5_hash(f"{st.st_mtime_ns}-{st.st_size}")
    except Exception:
      content_hash = md5_hash(transcript)
    if not FORCE_REINDEX:
      row = cur.execute("SELECT content_hash FROM indexed_files WHERE path=?", (transcript,)).fetchone()
      if row is not None and row[0] == content_hash:
        continue

    # parse VTT robustly
    try:
      if webvtt is not None:
        captions = [c.text for c in webvtt.read(transcript)]
        full_text = " ".join(captions)
      else:
        # fallback: strip timestamps and cue numbers
        lines = open(transcript, 'r', encoding='utf-8', errors='ignore').read().splitlines()
        cleaned = []
        for ln in lines:
          if re.match(r"^\d{2}:\d{2}:|^\d{2}:\d{2}:\d{2}|^NOTE|^WEBVTT$", ln):
            continue
          if re.match(r"^\d+$", ln.strip()):
            continue
          cleaned.append(ln)
        full_text = " ".join(cleaned)

    except Exception as e:
      print(f"Failed to parse VTT {transcript}: {e}")
      errs.append([transcript, vid_addr, str(e)])
      continue

    words = full_text.split()
    lines = [" ".join(words[i*tbs:min(len(words),(i+1)*tbs)]) for i in range(max(1, len(words)//tbs))]

    # prepare static vectors
    static_texts = [author or "", title or "", description or ""]
    static_vecs = encode_texts_with_cache(static_texts, model, cur, con)

    operations = []
    for idx, line in enumerate(lines):
      line_vec = encode_texts_with_cache([line], model, cur, con)[0]
      # weights [5,5,80]
      combined = [(static_vecs[0][i]*5 + static_vecs[1][i]*5 + line_vec[i]*80) / 90.0 for i in range(len(line_vec))]
      doc_id = md5_hash(f"{transcript}|{idx}|{line[:256]}")
      operations.append({"create": {"_index": TARGET_INDEX, "_id": doc_id}})
      operations.append({
        "title": f"{title} pt {str(1+idx)}/{str(len(lines))}",
        "text_vector": combined,
        "text": line,
        "description": description,
        "author": author
      })

    if len(operations) > 0:
      for chunk in chunk_operations(operations):
        try:
          res = es_bulk_with_retries(client, chunk, TARGET_INDEX)
          if isinstance(res, dict) and res.get('errors'):
            errs.append([transcript, vid_addr, 'partial_errors_in_bulk'])
        except Exception as e:
          print(f"Bulk indexing error (video {transcript}): {e}")
          errs.append([transcript, vid_addr, str(e)])

    # update indexed_files table
    try:
      mtime = int(Path(transcript).stat().st_mtime)
    except Exception:
      mtime = int(time.time())
    cur.execute("INSERT OR REPLACE INTO indexed_files(path, mtime, content_hash, indexed_at, doc_count) VALUES (?, ?, ?, datetime('now'), ?)",
                (transcript, mtime, content_hash, len(lines)))
    con.commit()
  except Exception as e:
    print(e)
    errs.append([item[0],item[1],str(e)])

try:
  # Refresh the physical target index first
  client.indices.refresh(index=TARGET_INDEX)
except Exception:
  pass

# If we performed an atomic reindex, swap the logical alias to the new index
if ATOMIC_REINDEX:
  try:
    # find previous indices behind the logical alias (if any)
    old_indices = []
    try:
      existing = client.indices.get_alias(name=LOGICAL_INDEX)
      old_indices = list(existing.keys())
    except Exception:
      old_indices = []

    actions = []
    for old in old_indices:
      if old != TARGET_INDEX:
        actions.append({"remove": {"index": old, "alias": LOGICAL_INDEX}})
    # add alias pointing to the new index
    actions.append({"add": {"index": TARGET_INDEX, "alias": LOGICAL_INDEX}})

    client.indices.update_aliases({"actions": actions})
    print(f"Alias '{LOGICAL_INDEX}' now points to {TARGET_INDEX}")

    # Optionally delete the old physical indices to reclaim space
    if DROP_OLD_INDEX and len(old_indices) > 0:
      for old in old_indices:
        if old != TARGET_INDEX:
          try:
            client.indices.delete(index=old)
            print(f"Deleted old index {old}")
          except Exception as e:
            print(f"Failed to delete old index {old}: {e}")
  except Exception as e:
    print(f"Alias swap failed: {e}")

print("error'd books/transcripts")
for item in errs:
  for field in item:
    print(field)
  print("\n\n")

# Cleanup easyocr readers (if any) to reduce leaked OS resources on some platforms
if easyocr is not None and isinstance(_easyocr_readers, dict):
  for k, r in list(_easyocr_readers.items()):
    try:
      if hasattr(r, 'close'):
        r.close()
    except Exception:
      pass
    try:
      del _easyocr_readers[k]
    except Exception:
      pass
  gc.collect()

