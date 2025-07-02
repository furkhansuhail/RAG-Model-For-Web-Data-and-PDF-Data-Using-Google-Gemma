import html
import hashlib
import requests
import json
import re
import os
import time
import textwrap
import numpy as np
import pandas as pd
import re
import spacy
import random
import sys

# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)



from langchain.schema import Document
from pathlib import Path
import matplotlib.pyplot as plt


# Data Cleanup
from langchain.schema import Document
from pathlib import Path
import requests, json, hashlib, re, html
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect  # pip install langdetect
from bs4 import BeautifulSoup
from spacy.lang.en import English


from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from langchain.schema import Document  # ✅ Correct LangChain Document import
from readability import Document as ReadabilityDocument  # ✅ Avoid name conflict
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import nltk
from nltk.tokenize import sent_tokenize  # You must have nltk downloaded





# Imports for FAISS and searches
import faiss
from rank_bm25 import BM25Okapi

# Pickle
import pickle

# To Calculate Runtimes
from time import perf_counter as timer

# To get google Search weblinks Links
from googlesearch import search


# To see the progress
from tqdm import tqdm




from readability import Document as ReadabilityDocument
from transformers import AutoConfig





# PDF Imports
import fitz  # PyMuPDF for reading PDFs
nltk.download('punkt')
from nltk.tokenize import sent_tokenize




# os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


# streamlit app
import streamlit as st
import streamlit.components.v1 as components



import sys
import types

# Patch torch.classes to prevent Streamlit from trying to introspect it
# Torch
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)

# Importing Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
try:
    from transformers import BitsAndBytesConfig
    import importlib.metadata

    # Check if bitsandbytes is installed
    importlib.metadata.version("bitsandbytes")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
except Exception as e:
    print(f"[Warning] 4-bit quantization disabled due to error: {e}")
    quantization_config = None

import unicodedata

# LLM
model_id = "google/gemma-7b-it"
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)
# modelId = "google/gemma-7b-it"
from transformers import GenerationConfig