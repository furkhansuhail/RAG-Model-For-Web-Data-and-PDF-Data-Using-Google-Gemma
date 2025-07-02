from ImportsForModel import *  # Assumes fitz, re, nltk, spacy, pandas, tqdm, streamlit, etc.
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []
# Hybrid RAG system: scraping, preprocessing, chunking, embedding, querying, and visualizing results in Streamlit.


class WEB_PDF_LLM_RAG_Application:
    def __init__(self, topic, number_results, mode, pdf_bytes=None, verbose=False, rag_search_type=None, file_name=None):
        os.makedirs("EmbeddingStorage", exist_ok=True)
        self.file_name = file_name
        # self.save_path_Weblinks_Embeddings = "EmbeddingStorage/WebLinks_EmbeddedData.pkl"
        # self.save_path_pdf_Embeddings = "EmbeddingStorage/PDF_EmbeddedData.pkl"
        self.save_path_Combined_Embeddings = "EmbeddingStorage/EmbeddedData.pkl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RAG Configuration
        self.topic = topic
        self.number_results = number_results
        self.mode = mode
        self.pdf_bytes = pdf_bytes  # ✅ Use the passed pdf_bytes instead of hardcoding file
        self.rag_search_type = rag_search_type
        self.verbose = verbose
        self.overlapSentences = 3

        self.pages_and_text_list_WebLinks = []
        self.pages_and_chunks_WebLinks = []
        self.pages_and_chunks_pdf = []
        self.search_results_combined = []
        self.search_method_used = ""
        self.first_chapter_page = None

        # LLM
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        self.model_id = model_id
        self.tokenizer = None
        self.llm_model = None
        self.pages_and_chunks_combined_List_Dict = []

    def SamplePrint(self, list_dict):
        test_weblink = random.sample(list_dict, k=1)
        df = pd.DataFrame(list_dict)
        print("🎲 Random Page Sample:")
        for key, value in test_weblink[0].items():
            print(f"{key}: {value}")
        print(df.describe().round(2))

    def ModelDriver(self):
        self.Data_Gathering_Processing()
        return self.LLM_Model_Setup()
        # query_system = self.LLMQuerySystem(save_path_Combined_Embeddings="EmbeddingStorage/CombinedData.pkl",
        #                               topic="What is insulin resistance?")
        # query_system.LLM_Model_Setup()


    def Data_Gathering_Processing(self, st_container=None, use_full_pdf=False):
        # _________________________________  Gather PDF + Web Data  _______________________________________________
        self.url_list = self.get_google_results(self.topic, self.number_results)
        print(f"🔗 URLs Fetched: {self.url_list}")

        Documents = self.scrape_and_clean_pages(self.url_list)
        print(f"📄 Scraped Pages: {len(Documents)}")

        self.pages_and_text_list_WebLinks = []
        chunk_char_len = 1200  # ~300 tokens assuming 4 chars per token
        chunk_overlap = 200  # slight overlap for context continuity

        for idx, doc in enumerate(Documents):
            raw_text = doc.get("text", "").strip()
            title = self.clean_for_llm(doc.get("title", "").strip())
            source = doc.get("source", "")
            # print(title, "|", source)
            # Clean the text before chunking
            cleaned_text = self.clean_for_llm(raw_text)

            # Split into overlapping chunks
            chunks = []
            start = 0
            while start < len(cleaned_text):
                chunk = cleaned_text[start:start + chunk_char_len]
                if len(chunk) < 100:  # skip too-short chunks
                    break
                chunks.append(chunk)
                start += chunk_char_len - chunk_overlap  # move with overlap

            for chunk_idx, chunk in enumerate(chunks):
                self.pages_and_text_list_WebLinks.append({
                    "page_number": f"{idx}_{chunk_idx}",
                    "page_char_count": len(chunk),
                    "page_word_count": len(chunk.split()),
                    "page_sentence_count_raw": len(chunk.split(". ")),
                    "page_token_count": len(chunk) / 4,
                    "text": chunk,
                    "source": source,
                    "title": title
                })

        self.pages_and_text_list_pdf = self.read_pdf_pages(start_page_number=41, pdf_bytes=self.pdf_bytes)


        # _______________________________  Sentencizing_NLP _____________________________________________________
        avg_sent_weblink, self.pages_and_text_list_WebLinks = self.Sentencizing_Chunking(
            self.pages_and_text_list_WebLinks, self.file_name)

        avg_sent_pdf, self.pages_and_text_list_pdf = self.Sentencizing_Chunking(
            self.pages_and_text_list_pdf, self.file_name)

        # # ________________________________  Splitting Chunks  ______________________________________________________
        print("****************************  Splitting Chunks  ******************************************************")
        # For WebLink data
        self.pages_and_chunks_WebLinks = self.Split_Chunks(
            self.pages_and_text_list_WebLinks, source_type="web", min_token_length=10
        )

        # For PDF data
        self.pages_and_chunks_pdf = self.Split_Chunks(
            self.pages_and_text_list_pdf, source_type="pdf", min_token_length=30
        )

        print("*******************************************************************************************************")
        print("________________________________  Embedding Chunks  ___________________________________________________")
        self.Combining_pages_chunks()
        # For PDF
        self.embed_chunks_universal(
            save_path=self.save_path_Combined_Embeddings,
            pages_and_chunks=self.pages_and_chunks_combined_List_Dict,
            source_type="pdf"
        )
        print("DATA Gathering Complete")



    def Combining_pages_chunks(self):
        seen_chunks = set()
        for chunk in self.pages_and_chunks_WebLinks + self.pages_and_chunks_pdf:
            chunk_text = chunk.get("sentence_chunk")
            if chunk_text not in seen_chunks:
                self.pages_and_chunks_combined_List_Dict.append(chunk)
                seen_chunks.add(chunk_text)


    def get_google_results(self, query: str, num_results: int) -> list:
        urls = []
        try:
            raw_results = search(query, num_results=num_results * 2)
            for result in raw_results:
                if "youtube.com" in result or "youtu.be" in result:
                    continue
                urls.append(result)
                if len(urls) >= num_results:
                    break
        except Exception as e:
            print(f"❌ Google Search failed: {e}")
        return urls


    def scrape_and_clean_pages(self, url_list):
        Path("scraped_texts").mkdir(exist_ok=True)
        seen_texts = set()
        nlp = English()
        nlp.add_pipe("sentencizer")

        def clean_text(text: str) -> str:
            text = re.sub(r'\s+', ' ', text)
            text = text.encode('ascii', errors='ignore').decode()
            text = re.sub(r'\.([A-Z])', r'. \1', text)
            boilerplate_keywords = ['cookie policy', 'terms of service', 'privacy policy', 'enable javascript', 'ad blocker', 'copyright', 'all rights reserved']
            for phrase in boilerplate_keywords:
                text = re.sub(rf'\b{re.escape(phrase)}\b', '', text, flags=re.IGNORECASE)
            lines = text.split('. ')
            if len(lines) > 10:
                most_common = max(set(lines), key=lines.count)
                if lines.count(most_common) > 3:
                    text = '. '.join([line for line in lines if line != most_common])
            return text.strip()

        documents = []
        for url in url_list:
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                raw_text = soup.get_text()
                formatted_text = clean_text(raw_text)
                if formatted_text not in seen_texts and len(formatted_text.split()) > 100:
                    seen_texts.add(formatted_text)
                    documents.append({
                        "text": formatted_text,
                        "source": url,
                        "title": soup.title.string if soup.title else "No Title"
                    })
            except Exception as e:
                print(f"❌ Failed to scrape {url}: {e}")
        return documents

    def clean_for_llm(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,;:!?()'\"-]", "", text)
        return text.strip()




    def find_first_chapter_page_auto_skip(self, st_container=None, use_full_pdf=False, pdf_bytes=None):
        if use_full_pdf:
            if st_container:
                st_container.info("📖 Showing full PDF (skipping chapter detection)")
            return 0

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        toc_keywords = ["table of contents", "contents", "index", "foreword", "preface", "introduction"]
        toc_pages = set()

        for page_number in range(min(10, len(doc))):
            text = doc[page_number].get_text()
            if any(kw in text.lower() for kw in toc_keywords):
                toc_pages.add(page_number)

        chapter_patterns = [
            r"^\s*chapter\s*0*1\b", r"^\s*chapter\s+one\b", r"^\s*1[\.\)]?\s+[A-Z]",
            r"^\s*I[\.\)]?\s+[A-Z]", r"^\s*第一章", r"^\s*capítulo\s+1",
        ]
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in chapter_patterns]

        matches = []
        for page_number in range(len(doc)):
            if page_number in toc_pages:
                continue
            text = doc[page_number].get_text()
            lines = text.split("\n")
            for line_number, line in enumerate(lines[:15]):
                stripped_line = line.strip()
                for pattern in compiled_patterns:
                    if pattern.match(stripped_line):
                        matches.append((page_number, line_number, stripped_line))

        doc.close()

        if matches:
            first_page = matches[0][0]
            print(f"✅ First Chapter Match on Page {first_page + 1}: \"{matches[0][2]}\"")
            if st_container:
                st_container.success(f"✅ First Chapter Match on Page {first_page + 1}")
            return first_page
        else:
            print("⚠️ Chapter 1 not confidently found, defaulting to page 1")
            if st_container:
                st_container.warning("⚠️ Chapter 1 not found. Using full PDF.")
            return 0

    def read_pdf_pages(self, start_page_number=0, pdf_bytes=None, pdf_path=None):
        """
        Reads a PDF from either bytes or file path and returns page-level data.
        """
        if pdf_bytes is not None and isinstance(pdf_bytes, bytes):
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        elif pdf_path is not None and isinstance(pdf_path, str):
            doc = fitz.open(pdf_path)
        else:
            raise ValueError("Provide either 'pdf_bytes' as bytes or 'pdf_path' as string.")

        pages_and_text = []
        for page_number, page in tqdm(enumerate(doc), desc="📄 Reading PDF Pages"):
            if page_number < start_page_number:
                continue
            text = page.get_text()
            pages_and_text.append({
                "page_number": page_number,
                "page_char_count": len(text),
                "page_word_count": len(text.split()),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
                "title": self.file_name, #or
                "source": self.file_name or pdf_path or "Unknown"
            })

        doc.close()
        return pages_and_text

    def Sentencizing_NLP(self, pages_and_text_list, filename=None, target_sentences=10):
        from spacy.lang.en import English
        from tqdm import tqdm
        import pandas as pd

        nlp = English()
        nlp.add_pipe("sentencizer")

        total_chunks = 0
        for item in tqdm(pages_and_text_list, desc=f"🧠 Sentencizing {filename or 'Data'}"):
            text = item.get("text", "").strip()
            if not text or len(text.split()) < 5:
                item["sentences"] = []
                item["sentence_chunks"] = []
                item["num_chunks"] = 0
                continue

            # Sentence segmentation
            doc = nlp(text)
            sentences = [str(sent).strip() for sent in doc.sents if sent.text.strip()]
            item["sentences"] = sentences
            item["page_sentence_count_spacy"] = len(sentences)

            # Chunking into lists of sentences
            sentence_chunks = []
            for i in range(0, len(sentences), target_sentences):
                chunk = sentences[i:i + target_sentences]
                if len(chunk) < 3:
                    continue
                sentence_chunks.append(chunk)

            item["sentence_chunks"] = sentence_chunks
            item["num_chunks"] = len(sentence_chunks)
            total_chunks += len(sentence_chunks)

        avg_chunk_sentences = int(total_chunks / len(pages_and_text_list)) if pages_and_text_list else 0
        return avg_chunk_sentences, pages_and_text_list




    def Sentencizing_Chunking(self, pages_and_text_list, filename=None, target_sentences=10):
        from spacy.lang.en import English
        from tqdm import tqdm
        import pandas as pd

        nlp = English()
        nlp.add_pipe("sentencizer")

        total_chunks = 0
        for item in tqdm(pages_and_text_list, desc=f"🧠 Sentencizing {filename or 'Data'}"):
            text = item.get("text", "").strip()
            if not text or len(text.split()) < 5:
                item["sentences"] = []
                item["sentence_chunks"] = []

                continue

            # Add page-level metadata if not already present
            item["page_char_count"] = len(text)
            item["page_word_count"] = len(text.split())
            item["page_sentence_count_raw"] = len(text.split(". "))
            item["page_token_count"] = len(text) / 4  # Rough token count estimate
            item["title"] = item.get("title")
            item["source"] = item.get("source") #self.file_name or filename or "Unknown"

            # Sentence segmentation
            doc = nlp(text)
            sentences = [str(sent).strip() for sent in doc.sents if sent.text.strip()]
            item["sentences"] = sentences
            item["page_sentence_count_spacy"] = len(sentences)

            # Chunking into sentence groups
            sentence_chunks = []
            for i in range(0, len(sentences), target_sentences):
                chunk = sentences[i:i + target_sentences]
                if len(chunk) < 3:
                    continue
                sentence_chunks.append(chunk)  # Append as list-of-sentences

            item["sentence_chunks"] = sentence_chunks
            item["num_chunks"] = len(sentence_chunks)
            total_chunks += len(sentence_chunks)

        avg_chunk_sentences = int(total_chunks / len(pages_and_text_list)) if pages_and_text_list else 0
        return avg_chunk_sentences, pages_and_text_list
        # print(pages_and_text_list)

    def Split_Chunks(self, pages_and_text_list, source_type="web", min_token_length=10):
        """
        Splits sentence chunks into flat records, suitable for embedding.
        Applies consistent cleaning and metadata preservation for both Web and PDF sources.

        Args:
            pages_and_text_list: List of pages with 'sentence_chunks' field.
            source_type: "web" or "pdf" to handle optional metadata fields like 'source' or 'text_path'.
            min_token_length: Minimum chunk size in estimated tokens to retain the chunk.

        Returns:
            filtered_chunks: List of clean chunk dictionaries that meet the token length threshold.
        """
        all_chunks = []

        for item in tqdm(pages_and_text_list, desc=f"📚 Splitting Chunks ({source_type})"):
            for sentence_chunk in item.get("sentence_chunks", []):
                chunk_dict = {}

                # Metadata retention (commonly shared)
                chunk_dict["page_number"] = item.get("page_number")

                if source_type == "web":
                    chunk_dict["source"] = item.get("source")
                    chunk_dict["title"] = item.get("title")
                    chunk_dict["text_path"] = item.get("text_path")

                if source_type == "pdf":
                    chunk_dict["source"] = item.get("source")
                    chunk_dict["title"] = item.get("title")
                    chunk_dict["text_path"] = "PDF"  # item.get("text_path")

                # Join sentences into one block of text
                joined = "".join(sentence_chunk).replace("  ", " ").strip()
                joined = re.sub(r'\.([A-Z])', r'. \1', joined)
                joined = re.sub(r'\s+', ' ', joined)
                joined = re.sub(r'[“”]', '"', joined)
                joined = re.sub(r"[’‘]", "'", joined)
                joined = re.sub(r"\s*–\s*", " - ", joined)

                token_count = len(joined) / 4  # Rough estimation

                chunk_dict["sentence_chunk"] = joined
                chunk_dict["chunk_char_count"] = len(joined)
                chunk_dict["chunk_word_count"] = len(joined.split())
                chunk_dict["chunk_token_count"] = token_count

                all_chunks.append(chunk_dict)

        # Convert to DataFrame for filtering and analysis
        stats_df = pd.DataFrame(all_chunks)

        # Debug: Print short chunk samples
        if not stats_df.empty and (stats_df["chunk_token_count"] <= min_token_length).any():
            short_samples = stats_df[stats_df["chunk_token_count"] <= min_token_length].sample(
                min(5, (stats_df["chunk_token_count"] <= min_token_length).sum())
            )
            for _, row in short_samples.iterrows():
                print(f"⚠️ Short Chunk ({row['chunk_token_count']:.1f} tokens): {row['sentence_chunk'][:100]}...")

        # Filter out short chunks
        filtered_chunks = stats_df[stats_df["chunk_token_count"] > min_token_length].to_dict(orient="records")

        print(f"✅ Total Valid Chunks: {len(filtered_chunks)}")
        return filtered_chunks

    def embed_chunks_universal(self, save_path, pages_and_chunks, source_type="pdf"):
        """
        Universal embedding function for both WebLink and PDF chunks.

        Args:
            save_path (str): Path to store the embeddings .pkl file
            pages_and_chunks (list): List of chunk dictionaries with 'sentence_chunk' key
            source_type (str): Either "pdf" or "web", used for hash handling and verbosity

        Returns:
            None
        """
        import hashlib

        def hash_chunk_text(text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        # Load existing data if present
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                existing_data = pickle.load(f)
            print(f"✅ Loaded existing embeddings from: {save_path}")

            if isinstance(existing_data.get("embeddings"), np.ndarray):
                existing_data["embeddings"] = existing_data["embeddings"].tolist()
        else:
            existing_data = {"chunks": [], "embeddings": []}

        # Hash-based deduplication
        existing_hashes = {
            hash_chunk_text(chunk["sentence_chunk"]) for chunk in existing_data["chunks"]
        }

        current_chunk_map = {
            hash_chunk_text(chunk["sentence_chunk"]): chunk for chunk in pages_and_chunks
        }

        new_hashes = set(current_chunk_map.keys()) - existing_hashes
        new_chunks = [current_chunk_map[h] for h in new_hashes]

        print(f"🆕 New {source_type} chunks to embed: {len(new_chunks)}")

        if not new_chunks:
            print("⚠️ No new chunks found.")
            return

        # Generate embeddings
        self.embedding_model.to(self.device)
        new_texts = [chunk["sentence_chunk"] for chunk in new_chunks]

        new_embeddings_tensor = self.embedding_model.encode(
            new_texts,
            show_progress_bar=True,
            convert_to_tensor=True
        ).to(self.device)

        # Combine with existing
        if existing_data["embeddings"]:
            existing_embeddings_tensor = torch.tensor(
                existing_data["embeddings"], dtype=torch.float32
            ).to(self.device)
        else:
            existing_embeddings_tensor = torch.empty(0, new_embeddings_tensor.shape[1]).to(self.device)

        combined_embeddings = torch.cat([existing_embeddings_tensor, new_embeddings_tensor], dim=0)
        combined_chunks = existing_data["chunks"] + new_chunks

        # Save updated data
        with open(save_path, "wb") as f:
            pickle.dump({
                "embeddings": combined_embeddings.cpu().numpy(),
                "chunks": combined_chunks
            }, f)

        print(f"💾 Embeddings updated and saved to {save_path}")
        print(
            f"📊 Previous: {len(existing_data['chunks'])} | Added: {len(new_chunks)} | Total: {len(combined_chunks)}")



    def TestLLM_Question_Format(self):
        input_text = "What are the macronutrients, and what roles do they play in the human body?"
        print(f"Input text:\n{input_text}")

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
             "content": input_text}
        ]

        # Apply the chat template
        prompt = self.tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,  # keep as raw text (not tokenized)
                                               add_generation_prompt=True)
        print(f"\nPrompt (formatted):\n{prompt}")

        # Tokenize the input text (turn it into numbers) and send it to GPU
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        print(f"Model input (tokenized):\n{input_ids}\n")

        # Generate outputs passed on the tokenized input
        # See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig
        outputs = self.llm_model.generate(**input_ids,
                                     max_new_tokens=256)  # define the maximum number of new tokens to create
        print(f"Model output (tokens):\n{outputs[0]}\n")

        # Decode the output tokens to text
        outputs_decoded = self.tokenizer.decode(outputs[0])
        print(f"Model output (decoded):\n{outputs_decoded}\n")



    # _____________________________________________ LLM SetUp ________________________________________________________
    def quantization_configuration_setup(self):
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_memory_bytes / (2 ** 30))
        print(f"Available GPU memory: {gpu_memory_gb} GB")
        use_quantization_config = None

        if gpu_memory_gb < 5.1:
            print(
                f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
            self.model_id = "google/gemma-2b-it"
        elif gpu_memory_gb < 8.1:
            print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
            use_quantization_config = True
            self.model_id = "google/gemma-2b-it"
        elif gpu_memory_gb < 19.0:
            print(
                f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
            use_quantization_config = False
            self.model_id = "google/gemma-2b-it"
        else:
            print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 7B in 4-bit or float16 precision.")
            use_quantization_config = False
            self.model_id = "google/gemma-7b-it"

        print(f"use_quantization_config set to: {use_quantization_config}")
        print(f"model_id set to: {self.model_id}")
        return use_quantization_config

    def get_model_num_params(self, model: torch.nn.Module):
        return sum([param.numel() for param in model.parameters()])

    def get_model_mem_size(self, model: torch.nn.Module):
        mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
        mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        model_mem_bytes = mem_params + mem_buffers
        model_mem_mb = model_mem_bytes / (1024 ** 2)
        model_mem_gb = model_mem_bytes / (1024 ** 3)
        return {"model_mem_bytes": model_mem_bytes, "model_mem_mb": round(model_mem_mb, 2),
                "model_mem_gb": round(model_mem_gb, 2)}

    def load_combined_embeddings(self):
        with open(self.save_path_Combined_Embeddings, "rb") as f:
            data = pickle.load(f)

        chunks = data["chunks"]
        embeddings = data["embeddings"]
        df = pd.DataFrame(chunks)
        df["embedding"] = list(embeddings)
        self.pages_and_chunks_combined = df.to_dict(orient="records")
        embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(self.device)
        return self.pages_and_chunks_combined, embeddings_tensor

    def retrieve_relevant_resources(self, query, embeddings, n_resources_to_return=5):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        start_time = timer()
        dot_scores = util.dot_score(query_embedding, embeddings)[0]
        end_time = timer()
        print(f"⚙️ Scoring time: {end_time - start_time:.2f}s")

        scores, indices = torch.topk(dot_scores, k=n_resources_to_return)
        context_items = [self.pages_and_chunks_combined[i] for i in indices]
        prompt = self.prompt_formatter(query=query, context_items=context_items)

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm_model.generate(
            **input_ids,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            max_new_tokens=2048,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_answer = output_text.replace(prompt, "").strip()

        print(f"\n🧾 Final Answer:\n{final_answer}")
        return scores, indices, final_answer

    def prompt_formatter(self, query, context_items):
        context_blocks = "\n".join([f"- {item['sentence_chunk'].strip()}" for item in context_items])

        prompt_template = f"""You are a knowledgeable assistant helping to answer questions based on a provided context.
Please analyze the information below and answer the user’s query as thoroughly as possible.

Context:
{context_blocks}

Instructions:
1. Extract and consider only the most relevant parts of the context.
2. Formulate a comprehensive answer to the user query.
3. Avoid repeating the query or including unnecessary formatting.

User Query:
{query}

Answer:"""

        chat_format = [{"role": "user", "content": prompt_template}]
        return self.tokenizer.apply_chat_template(chat_format, tokenize=False, add_generation_prompt=True)

    def LLM_PDF_WEB_Query_Search(self, query, similarity_method="llm"):
        embeddings_path = self.save_path_Combined_Embeddings
        chunks, embeddings = self.load_combined_embeddings()
        if not chunks or embeddings.numel() == 0:
            print("❌ No embeddings available.")
            return

        scores, indices, final_answer = self.retrieve_relevant_resources(query=query, embeddings=embeddings)
        results = []
        for i, score in zip(indices, scores):
            chunk = chunks[i]
            results.append({
                "text": chunk.get("sentence_chunk", ""),
                "score": score.item(),
                "source": f"WebLink: {chunk.get('source', '')}" if chunk.get("source") else "PDF Page",
                "metadata": {k: v for k, v in chunk.items() if k != "sentence_chunk"}
            })

        print("🧠 Top Results:")
        for r in results:
            print(f"- {r['source']}: {r['text'][:100]}...")

        return {
            "answer": final_answer,
            "top_chunks": results,
            "query": query,
            "search_method": "LLM"
        }

    def LLM_Model_Setup(self):
        use_quant = True if "2b" in self.model_id else False

        try:
            if torch.cuda.get_device_capability(0)[0] >= 8:
                import flash_attn
                attn_implementation = "flash_attention_2"
            else:
                attn_implementation = "sdpa"
        except ImportError:
            print("⚠️ FlashAttention 2 not installed. Falling back to SDPA.")
            attn_implementation = "sdpa"

        print(f"⚡ Attention type: {attn_implementation}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.float16) if use_quant else None

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            low_cpu_mem_usage=False,
            attn_implementation=attn_implementation
        )

        if not use_quant:
            self.llm_model.to(self.device)

        print(f"✅ Model loaded: {self.model_id}")
        return self.llm_model

