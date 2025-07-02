from ImportsForModel import *  # Assumes fitz, re, nltk, spacy, pandas, tqdm, streamlit, etc.
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []
# Hybrid RAG system: scraping, preprocessing, chunking, embedding, querying, and visualizing results in Streamlit.


class WEB_PDF_RAG_Application:
    def __init__(self, topic, number_results, mode, pdf_bytes=None, verbose=False, rag_search_type=None, file_name=None):
        os.makedirs("EmbeddingStorage", exist_ok=True)
        self.file_name = file_name
        self.save_path_Weblinks_Embeddings = "EmbeddingStorage/WebLinks_EmbeddedData.pkl"
        self.save_path_pdf_Embeddings = "EmbeddingStorage/PDF_EmbeddedData.pkl"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.topic = topic
        self.number_results = number_results
        self.mode = mode
        self.pdf_bytes = pdf_bytes
        self.rag_search_type = rag_search_type
        self.verbose = verbose

        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        self.overlapSentences = 3

        self.pages_and_text_list_WebLinks = []
        self.pages_and_chunks_WebLinks = []
        self.pages_and_chunks_pdf = []
        self.search_results_combined = []
        self.search_method_used = ""
        self.first_chapter_page = None

    def text_formatter(self, text: str) -> str:
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

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

    def read_pdf_pages(self, start_page_number=0, pdf_bytes=None):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_and_text = []
        for page_number, page in tqdm(enumerate(doc), desc="Reading PDF Pages"):
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
                'title': 'PDF File',
                'source': self.file_name
            })
        doc.close()
        return pages_and_text

    def clean_for_llm(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s.,;:!?()'\"-]", "", text)
        return text.strip()

    def Sentencizing_NLP(self, pages_and_text_list, filename=None, target_sentences=10):
        nlp = English()
        nlp.add_pipe("sentencizer")

        cleaned_pages = []
        seen_hashes = set()

        for item in tqdm(pages_and_text_list, desc=f"🧠 Sentencizing {filename or 'Data'}"):
            text = item.get("text", "").strip()
            if not text or len(text.split()) < 5:
                continue

            text_hash = hash(text)
            if text_hash in seen_hashes:
                continue
            seen_hashes.add(text_hash)

            sentences = [str(sent).strip() for sent in nlp(text).sents if len(str(sent).strip()) > 0]

            for i in range(0, len(sentences), target_sentences):
                chunk = sentences[i:i + target_sentences]
                if len(chunk) < 3:
                    continue

                chunk_text = " ".join(chunk)

                chunk_data = {
                    "sentence_chunk": chunk_text,
                    "sentences": chunk,
                    "chunk_sentence_count": len(chunk),
                    "chunk_char_count": len(chunk_text),
                    "chunk_word_count": len(chunk_text.split()),
                    "chunk_token_count": len(chunk_text) / 4
                }

                metadata = {k: v for k, v in item.items() if k not in ["text", "sentences"]}
                chunk_data.update(metadata)
                cleaned_pages.append(chunk_data)

        mean_sent_count = int(pd.DataFrame(cleaned_pages)["chunk_sentence_count"].mean()) if cleaned_pages else 0
        return mean_sent_count, cleaned_pages

    def Chunking_NLP(self, pages_and_text_list, average_sentences=10, overlap_sentences=3, source_type="web"):
        chunk_size = average_sentences
        stride = chunk_size - overlap_sentences

        pages_with_chunks = []
        all_chunks = []

        def split_with_overlap(sentences, chunk_size, stride):
            return [sentences[i:i + chunk_size] for i in range(0, len(sentences), stride) if
                    len(sentences[i:i + chunk_size]) >= 3]

        def split_without_overlap(sentences, chunk_size):
            return [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size) if
                    len(sentences[i:i + chunk_size]) >= 3]

        for item in tqdm(pages_and_text_list, desc=f"🔗 Chunking ({source_type})"):
            sentences = item.get("sentences", [])
            chunks = split_with_overlap(sentences, chunk_size,
                                        stride) if source_type == "pdf" else split_without_overlap(sentences,
                                                                                                   chunk_size)

            item["sentence_chunks"] = chunks
            item["num_chunks"] = len(chunks)
            pages_with_chunks.append(item)

            for chunk in chunks:
                joined = " ".join(chunk).strip()
                joined = re.sub(r'\.([A-Z])', r'. \1', joined)
                joined = re.sub(r'\s+', ' ', joined)
                joined = re.sub(r'[“”]', '"', joined)
                joined = re.sub(r"[’‘]", "'", joined)
                joined = re.sub(r"\s*–\s*", " - ", joined)

                token_count = len(joined) / 4
                if token_count < 30:
                    continue

                chunk_dict = {
                    "sentence_chunk": joined,
                    "chunk_char_count": len(joined),
                    "chunk_word_count": len(joined.split()),
                    "chunk_token_count": token_count,
                }

                for key in ["page_number", "source", "title"]:
                    if key in item:
                        chunk_dict[key] = item[key]

                all_chunks.append(chunk_dict)

        print(f"✅ Total Chunks: {len(all_chunks)}")
        return pages_with_chunks, all_chunks

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
                    chunk_dict["text_path"] = "PDF" #item.get("text_path")

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
        embedding_model.to(self.device)
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
        print(f"📊 Previous: {len(existing_data['chunks'])} | Added: {len(new_chunks)} | Total: {len(combined_chunks)}")


    def Data_Gathering_Processing(self, rag_search_type="dot product", st_container=None, use_full_pdf=False):
        print("✅ Correct WEB_PDF_RAG_Application class is being used.")
        print(f"🔍 Starting search using: {rag_search_type}")


        # _________________________________  Gather PDF + Web Data  _______________________________________________

        self.url_list = self.get_google_results(self.topic, self.number_results)
        print(f"🔗 URLs Fetched: {self.url_list}")

        Documents = self.scrape_and_clean_pages(self.url_list)
        print(f"📄 Scraped Pages: {len(Documents)}")

        self.pages_and_text_list_WebLinks = []
        for idx, doc in enumerate(Documents):
            raw_text = doc.get("text", "").strip()
            title = doc.get("title", "").strip()

            text = self.clean_for_llm(raw_text)
            title = self.clean_for_llm(title)

            self.pages_and_text_list_WebLinks.append({
                "page_number": idx,
                "page_char_count": len(text),
                "page_word_count": len(text.split()),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,
                "text": text,
                "source": doc.get("source", ""),
                "title": title
            })


        self.first_chapter_page = self.find_first_chapter_page_auto_skip(
            st_container=st_container, use_full_pdf=use_full_pdf, pdf_bytes=self.pdf_bytes
        )

        self.pages_and_text_list_pdf = self.read_pdf_pages(
            start_page_number=self.first_chapter_page, pdf_bytes=self.pdf_bytes
        )

        # _______________________________  Sentencizing_NLP _____________________________________________________
        avg_sent_weblink, self.pages_and_text_list_WebLinks = self.Sentencizing_NLP(
            self.pages_and_text_list_WebLinks, self.file_name)

        avg_sent_pdf, self.pages_and_text_list_pdf = self.Sentencizing_NLP(
            self.pages_and_text_list_pdf, self.file_name)

        #________________________________  Chunking  ____________________________________________________________
        _, self.pages_and_chunks_WebLinks = self.Chunking_NLP(
            self.pages_and_text_list_WebLinks, avg_sent_weblink, source_type="web")

        _, self.pages_and_chunks_pdf = self.Chunking_NLP(
            self.pages_and_text_list_pdf, avg_sent_pdf, source_type="pdf")
        # print("****************************  CHUNING ***********************************************************")
        # # ✅ Print sample chunks
        # print("\n📌 Sample WebLink Chunk:")
        # WebLink_KeyList = []
        # if self.pages_and_chunks_WebLinks:
        #     sample_web_chunk = self.pages_and_chunks_WebLinks[0]
        #     for k, v in sample_web_chunk.items():
        #         print(f"{k}:{v}")
        #         WebLink_KeyList.append(k)
        # print("WebLink:", WebLink_KeyList)
        #
        # print("\n📌 Sample PDF Chunk:")
        # PDFLink_KeyList = []
        # if self.pages_and_chunks_pdf:
        #     sample_pdf_chunk = self.pages_and_chunks_pdf[0]
        #     for k, v in sample_pdf_chunk.items():
        #         print(f"{k}: {v}")
        #         PDFLink_KeyList.append(k)
        # print("PDFLink:", PDFLink_KeyList)

        # ________________________________  Splitting Chunks  ______________________________________________________
        # print("****************************  Splitting Chunks  ******************************************************")
        # For WebLink data
        self.pages_and_chunks_WebLinks = self.Split_Chunks(
            self.pages_and_text_list_WebLinks, source_type="web", min_token_length=10
        )

        # For PDF data
        self.pages_and_chunks_pdf = self.Split_Chunks(
            self.pages_and_text_list_pdf, source_type="pdf", min_token_length=30
        )

        # # ✅ Print sample chunks
        # print("\n📌 Sample WebLink Chunk:")
        # WebLink_KeyList = []
        # if self.pages_and_chunks_WebLinks:
        #     sample_web_chunk = self.pages_and_chunks_WebLinks[0]
        #     for k, v in sample_web_chunk.items():
        #         print(f"{k}:{v}")
        #         WebLink_KeyList.append(k)
        # print("WebLink:", WebLink_KeyList)
        #
        # print("\n📌 Sample PDF Chunk:")
        # PDFLink_KeyList = []
        # if self.pages_and_chunks_pdf:
        #     sample_pdf_chunk = self.pages_and_chunks_pdf[0]
        #     for k, v in sample_pdf_chunk.items():
        #         print(f"{k}: {v}")
        #         PDFLink_KeyList.append(k)
        # print("PDFLink:", PDFLink_KeyList)

        # print("*******************************************************************************************************")
        # # ________________________________  Embedding Chunks  ______________________________________________________
        # For PDF
        self.embed_chunks_universal(
            save_path=self.save_path_pdf_Embeddings,
            pages_and_chunks=self.pages_and_chunks_pdf,
            source_type="pdf"
        )

        # For WebLinks
        self.embed_chunks_universal(
            save_path=self.save_path_Weblinks_Embeddings,
            pages_and_chunks=self.pages_and_chunks_WebLinks,
            source_type="web"
        )




        return self.SearchModuleSetup()


    # def SearchModuleSetup(self):
    #     results, method = self.Semantic_Rag_DotProduct_Search(
    #         pdf_path=self.save_path_pdf_Embeddings,
    #         web_path=self.save_path_Weblinks_Embeddings,
    #         query=self.topic,
    #         similarity_method=self.rag_search_type
    #     )
    #     self.search_results_combined = results
    #     self.search_method_used = method
    #     return results, method

    def SearchModuleSetup(self):
        query = self.topic
        pdf_path = self.save_path_pdf_Embeddings
        web_path = self.save_path_Weblinks_Embeddings

        # Load chunks for methods requiring only chunks
        chunks_pdf, _ = self.load_pdf_embeddings(pdf_path)
        chunks_web, _ = self.load_pdf_embeddings(web_path)

        rag_type = (self.rag_search_type or "dot product").lower().strip()

        if rag_type in ["dot product", "cosine"]:
            results, method = self.Semantic_Rag_DotProduct_Search(
                pdf_path=pdf_path,
                web_path=web_path,
                query=query,
                similarity_method=rag_type
            )

        elif rag_type == "euclidean":
            results, method = self.Semantic_Rag_Euclidean_Search(
                pdf_path=pdf_path,
                web_path=web_path,
                query=query
            )

        elif rag_type == "faiss":
            results, method = self.Semantic_Rag_FAISS_Search(
                pdf_path=pdf_path,
                web_path=web_path,
                query=query
            )

        elif rag_type in ["hybrid", "bm25"]:
            results, method = self.Hybrid_BM25_Embedding_Search(
                pdf_chunks=chunks_pdf,
                web_chunks=chunks_web,
                query=query
            )

        elif rag_type in ["cross", "cross encoder", "cross-encoder"]:
            results, method = self.CrossEncoder_Reranking_Search(
                chunks_pdf=chunks_pdf,
                chunks_web=chunks_web,
                query=query
            )

        else:
            print(f"❌ Unknown search type: {rag_type}. Defaulting to dot product.")
            results, method = self.Semantic_Rag_DotProduct_Search(
                pdf_path=pdf_path,
                web_path=web_path,
                query=query,
                similarity_method="dot product"
            )

        self.search_results_combined = results
        self.search_method_used = method
        return results, method

    def load_pdf_embeddings(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        chunks = data["chunks"]
        embeddings = data["embeddings"]

        # Convert to tensor if not already
        embeddings_tensor = torch.tensor(
            np.array([
                np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else np.array(x)
                for x in embeddings
            ]),
            dtype=torch.float32
        ).to(device)

        return chunks, embeddings_tensor

    def prepare_combined_embeddings(self, pdf_chunks, web_chunks):
        combined_chunks = []
        all_embeddings = []

        for chunk in pdf_chunks:
            chunk["source_type"] = "pdf"
            all_embeddings.append(np.array(chunk["embedding"]))
            combined_chunks.append(chunk)

        for chunk in web_chunks:
            chunk["source_type"] = "web"
            # Fix: get source_url from chunk (not chunk["chunk"])
            chunk["source_url"] = chunk.get("source", "Unknown Source")
            all_embeddings.append(np.array(chunk["embedding"]))
            combined_chunks.append(chunk)

        embeddings_tensor = torch.tensor(np.array(all_embeddings), dtype=torch.float32).to(device)
        return combined_chunks, embeddings_tensor

    def Semantic_Rag_DotProduct_Search(self, pdf_path, web_path, query, similarity_method="dot_product"):
        chunks_pdf, embeddings_pdf = self.load_pdf_embeddings(pdf_path)
        chunks_web, embeddings_web = self.load_pdf_embeddings(web_path)

        if not chunks_pdf and not chunks_web:
            print("⚠️ No data found in PDF or Web embeddings.")
            return {
                "pdf_results": [],
                "web_results": [],
                "combined_results": []
            }, "No Results - Empty Embeddings"

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)

        def get_top_results(chunks, embeddings, label, mode):
            if not chunks or embeddings.numel() == 0:
                return []

            if similarity_method == "dot product":
                scores = util.dot_score(query_embedding, embeddings)[0]
            elif similarity_method == "cosine":
                scores = util.cos_sim(query_embedding, embeddings)[0]

            else:
                print("❌ Unsupported similarity method.")
                return []

            top_k = min(5, len(scores))
            top_indices = torch.topk(scores, k=top_k).indices.tolist()

            results = []
            for i in top_indices:
                try:
                    chunk = chunks[i]
                    score = scores[i].item()
                    text = chunk.get("sentence_chunk", "")
                    page_number = chunk.get("page_number")
                    source_url = chunk.get("source", "")
                    title = chunk.get("title", "")
                    text_path = chunk.get("text_path", "")
                    metadata = {k: v for k, v in chunk.items() if k != "sentence_chunk"}

                    # Determine formatted source
                    if mode == "Pdf":
                        final_source = f"PDF Page {page_number}" if page_number is not None else "PDF Source"
                    elif mode == "Web":
                        final_source = f"WebLink: {source_url}" if source_url else "Web Source"
                    elif mode == "Combined":
                        parts = []
                        if page_number is not None:
                            parts.append(f"PDF Page {page_number}")
                        if source_url:
                            parts.append(f"WebLink: {source_url}")
                        final_source = " + ".join(parts) if parts else "Unknown Source"
                    else:
                        final_source = "Unknown Source"

                    results.append({
                        "text": text,
                        "score": score,
                        "source": final_source,
                        "metadata": metadata
                    })

                except Exception as e:
                    print(f"⚠️ Error at index {i}: {e}")
            # print(results)
            return results

        # Individual searches
        pdf_results = get_top_results(chunks_pdf, embeddings_pdf, "PDF Source", mode="Pdf")
        # print(pdf_results)
        web_results = get_top_results(chunks_web, embeddings_web, "WebLink Source", mode="Web")
        # print(web_results)
        # Combined search
        if chunks_pdf and chunks_web:
            combined_chunks = chunks_pdf + chunks_web
            combined_embeddings = torch.cat([embeddings_pdf, embeddings_web], dim=0)
        else:
            combined_chunks = chunks_pdf or chunks_web
            combined_embeddings = embeddings_pdf if chunks_pdf else embeddings_web

        combined_results = get_top_results(combined_chunks, combined_embeddings, "Combined Source", mode="Combined")

        return {
            "pdf_results": pdf_results,
            "web_results": web_results,
            "combined_results": combined_results
        }, f"{similarity_method.replace('_', ' ').title()} Search (Combined)"

    def Semantic_Rag_Euclidean_Search(self, pdf_path, web_path, query):
        from sklearn.metrics.pairwise import euclidean_distances

        chunks_pdf, embeddings_pdf = self.load_pdf_embeddings(pdf_path)
        chunks_web, embeddings_web = self.load_pdf_embeddings(web_path)

        if not chunks_pdf and not chunks_web:
            print("⚠️ No data found.")
            return {"pdf_results": [], "web_results": [], "combined_results": []}, "No Results"

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

        def get_top(chunks, embeddings, mode):
            if not chunks:
                return []
            distances = euclidean_distances(query_embedding, embeddings.cpu().numpy())[0]
            top_k = min(5, len(distances))
            indices = distances.argsort()[:top_k]
            return [
                {
                    "text": chunks[i].get("sentence_chunk", ""),
                    "score": float(distances[i]),
                    "source": f"{'PDF Page ' + str(chunks[i].get('page_number')) if mode == 'Pdf' else 'WebLink: ' + chunks[i].get('source', '')}",
                    "metadata": {k: v for k, v in chunks[i].items() if k != "sentence_chunk"}
                } for i in indices
            ]

        return {
            "pdf_results": get_top(chunks_pdf, embeddings_pdf, "Pdf"),
            "web_results": get_top(chunks_web, embeddings_web, "Web"),
            "combined_results": get_top(chunks_pdf + chunks_web, torch.cat([embeddings_pdf, embeddings_web], dim=0),
                                        "Combined")
        }, "Euclidean Distance Search"

    def Semantic_Rag_FAISS_Search(self, pdf_path, web_path, query):
        import faiss
        chunks_pdf, embeddings_pdf = self.load_pdf_embeddings(pdf_path)
        chunks_web, embeddings_web = self.load_pdf_embeddings(web_path)

        if not chunks_pdf and not chunks_web:
            print("⚠️ No data found.")
            return {"pdf_results": [], "web_results": [], "combined_results": []}, "No Results"

        query_embedding = self.embedding_model.encode(query).astype('float32').reshape(1, -1)

        def get_top(chunks, embeddings, mode):
            if not chunks:
                return []
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.cpu().numpy())
            D, I = index.search(query_embedding, k=min(5, len(chunks)))
            return [
                {
                    "text": chunks[i].get("sentence_chunk", ""),
                    "score": float(D[0][rank]),
                    "source": f"{'PDF Page ' + str(chunks[i].get('page_number')) if mode == 'Pdf' else 'WebLink: ' + chunks[i].get('source', '')}",
                    "metadata": {k: v for k, v in chunks[i].items() if k != "sentence_chunk"}
                } for rank, i in enumerate(I[0])
            ]

        return {
            "pdf_results": get_top(chunks_pdf, embeddings_pdf, "Pdf"),
            "web_results": get_top(chunks_web, embeddings_web, "Web"),
            "combined_results": get_top(chunks_pdf + chunks_web, torch.cat([embeddings_pdf, embeddings_web], dim=0),
                                        "Combined")
        }, "FAISS ANN Search"

    def Hybrid_BM25_Embedding_Search(self, pdf_chunks, web_chunks, query):
        from rank_bm25 import BM25Okapi

        all_chunks = pdf_chunks + web_chunks
        corpus = [chunk.get("sentence_chunk", "") for chunk in all_chunks]
        tokenized_corpus = [text.split() for text in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_scores = bm25.get_scores(query.split())
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]

        # Rerank top BM25 with embedding similarity
        top_chunks = [all_chunks[i] for i in top_bm25_indices]
        top_texts = [chunk["sentence_chunk"] for chunk in top_chunks]
        top_embeddings = self.embedding_model.encode(top_texts, convert_to_tensor=True).to(self.device)
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
        scores = util.dot_score(query_embedding, top_embeddings)[0]

        top_k = min(5, len(scores))
        top_indices = torch.topk(scores, k=top_k).indices.tolist()

        results = []
        for i in top_indices:
            chunk = top_chunks[i]
            results.append({
                "text": chunk.get("sentence_chunk", ""),
                "score": float(scores[i]),
                "source": chunk.get("source", ""),
                "metadata": {k: v for k, v in chunk.items() if k != "sentence_chunk"}
            })

        return {"combined_results": results}, "Hybrid BM25 + Embedding Search"

    def CrossEncoder_Reranking_Search(self, chunks_pdf, chunks_web, query,
                                      model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder(model_name)
        combined_chunks = chunks_pdf + chunks_web
        texts = [chunk.get("sentence_chunk", "") for chunk in combined_chunks]
        pairs = [(query, text) for text in texts]
        scores = cross_encoder.predict(pairs)

        top_k = min(5, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in top_indices:
            chunk = combined_chunks[i]
            results.append({
                "text": chunk.get("sentence_chunk", ""),
                "score": float(scores[i]),
                "source": chunk.get("source", ""),
                "metadata": {k: v for k, v in chunk.items() if k != "sentence_chunk"}
            })

        return {"combined_results": results}, "Cross-Encoder Reranked Search"


# ========== STREAMLIT RUNNER ==========

# def shutdown_streamlit():
#     st.warning("Shutting down the Streamlit app...")
#     os._exit(0)
#
# def run_streamlit_app():
#     file_name = ""
#     st.set_page_config(page_title="Hybrid RAG Search", layout="wide")
#     st.title("[Google + PDF] Semantic RAG with Google Gemma")
#
#     topic = st.text_input("Enter your query:")
#     verbose = st.toggle("Verbose Answer", value=False)
#     number_results = st.slider("Number of URLs to search", 3, 20, 5)
#     uploaded_file = st.file_uploader("Upload PDF for knowledge base", type="pdf")
#     use_full_pdf = st.checkbox("Use entire PDF (skip Chapter 1 detection)", value=False)
#
#     if uploaded_file is not None:
#         file_name = uploaded_file.name
#
#     search_mode = st.selectbox(
#         "Choose Search Mode",
#         ["google", "pdf only", "pdf + google search", "llm", "llm + pdf", "llm + google search", "llm + google search + pdf"]
#     )
#
#     rag_search_type = st.selectbox(
#         "Choose RAG Search Method",
#         ["dot product", "cosine", "euclidean", "faiss", "hybrid", "cross_encoder"]
#     )
#
#     requires_pdf = "pdf" in search_mode
#     disable_run = requires_pdf and not uploaded_file
#     run_button = st.button("Run Query", disabled=disable_run)
#
#     try:
#         if run_button and (topic or uploaded_file):
#             pdf_bytes = uploaded_file.read() if (uploaded_file and requires_pdf) else None
#
#             webapp = WEB_PDF_RAG_Application(
#                 topic=topic or "N/A",
#                 number_results=number_results,
#                 mode=search_mode,
#                 pdf_bytes=pdf_bytes,
#                 verbose=verbose,
#                 rag_search_type=rag_search_type,
#                 file_name=file_name
#             )
#
#             with st.spinner("Running pipeline..."):
#                 results, search_method = None, None
#
#                 if "pdf + google search" in search_mode and pdf_bytes:
#                     if rag_search_type == "dot product":
#                         results, search_method = webapp.Semantic_Rag_DotProduct_Search(
#                             webapp.save_path_pdf_Embeddings,
#                             webapp.save_path_Weblinks_Embeddings,
#                             topic,
#                             similarity_method="dot product"
#                         )
#                     elif rag_search_type == "cosine":
#                         results, search_method = webapp.Semantic_Rag_DotProduct_Search(
#                             webapp.save_path_pdf_Embeddings,
#                             webapp.save_path_Weblinks_Embeddings,
#                             topic,
#                             similarity_method="cosine"
#                         )
#                     elif rag_search_type == "euclidean":
#                         results, search_method = webapp.Semantic_Rag_Euclidean_Search(
#                             webapp.save_path_pdf_Embeddings,
#                             webapp.save_path_Weblinks_Embeddings,
#                             topic
#                         )
#                     elif rag_search_type == "faiss":
#                         results, search_method = webapp.Semantic_Rag_FAISS_Search(
#                             webapp.save_path_pdf_Embeddings,
#                             webapp.save_path_Weblinks_Embeddings,
#                             topic
#                         )
#                     elif rag_search_type == "hybrid":
#                         chunks_pdf, _ = webapp.load_pdf_embeddings(webapp.save_path_pdf_Embeddings)
#                         chunks_web, _ = webapp.load_pdf_embeddings(webapp.save_path_Weblinks_Embeddings)
#                         results, search_method = webapp.Hybrid_BM25_Embedding_Search(
#                             chunks_pdf, chunks_web, topic
#                         )
#                     elif rag_search_type == "cross_encoder":
#                         chunks_pdf, _ = webapp.load_pdf_embeddings(webapp.save_path_pdf_Embeddings)
#                         chunks_web, _ = webapp.load_pdf_embeddings(webapp.save_path_Weblinks_Embeddings)
#                         results, search_method = webapp.CrossEncoder_Reranking_Search(
#                             chunks_pdf, chunks_web, topic
#                         )
#
#             if "pdf" in search_mode and hasattr(webapp, "pages_and_text_list_pdf"):
#                 with st.expander("View Extracted PDF Pages", expanded=False):
#                     for i, page in enumerate(webapp.pages_and_text_list_pdf):
#                         if isinstance(page, dict):
#                             st.markdown(f"### Page {page.get('page_number', i) + 1}")
#                             st.text(page.get("text", ""))
#
#             if results and isinstance(results, dict):
#                 st.success(f"Search completed using {search_method}.")
#                 for label, result_list in [
#                     (" Top 5 Results from PDF", results.get("pdf_results", [])),
#                     (" Top 5 Results from Web", results.get("web_results", [])),
#                     (" Top 5 Results from Combined", results.get("combined_results", [])),
#                 ]:
#                     st.markdown(f"### {label}")
#                     if not result_list:
#                         st.info("No results found.")
#                         continue
#
#                     for i, res in enumerate(result_list):
#                         if not isinstance(res, dict):
#                             st.warning(f" Skipped invalid result: {res}")
#                             continue
#
#                         score = f"{res.get('score', 0):.4f}" if isinstance(res.get('score'), (int, float)) else "N/A"
#                         st.markdown(f"**Result {i + 1}**")
#                         st.markdown(f"**Score:** `{score}`")
#
#                         source_text = res.get("source", "")
#                         if "WebLink:" in source_text and "http" in source_text:
#                             parts = source_text.split("WebLink:")
#                             before_link = parts[0].strip(" +")
#                             url = parts[1].strip()
#                             st.markdown(f"**Source:** {before_link} + [WebLink]({url})  \n`{url}`")
#                         elif "http" in source_text:
#                             url = source_text.strip()
#                             st.markdown(f"**Source Link:** [{url}]({url})  \n`{url}`")
#                         else:
#                             st.markdown(f"**Source:** {source_text}")
#
#                         if verbose:
#                             st.markdown("**Full Metadata:**")
#                             st.json(res.get("metadata", {}))
#                             st.markdown("**Full Chunk Text:**")
#                             st.text(res.get("text", ""))
#                         else:
#                             snippet = res.get("text", "")[:1024]
#                             st.markdown("**Text Snippet:**")
#                             st.markdown(f"> {snippet}{'...' if len(snippet) >= 500 else ''}")
#                         st.markdown("---")
#
#     except Exception as e:
#         st.error(f"Error during execution: {e}")
#         st.exception(e)
#
#     st.markdown("---")
#     if st.button("Exit App"):
#         shutdown_streamlit()
#
# if __name__ == "__main__":
#     run_streamlit_app()

