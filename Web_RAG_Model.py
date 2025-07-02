from ImportsForModel import *  # Assumes fitz, re, nltk, spacy, pandas, tqdm, streamlit, etc.

class WEB_RAG_Application:
    def __init__(self, topic, number_results, mode, pdf_bytes=None, verbose=False):
        os.makedirs("EmbeddingStorage", exist_ok=True)
        self.save_path_Weblinks = "EmbeddingStorage/WebLinks_EmbeddedData.pkl"
        self.device = device
        self.topic = topic
        self.number_results = number_results
        self.mode = mode
        self.pdf_bytes = pdf_bytes
        self.pages_and_text_list = []
        self.pages_and_chunks_WebLinks = []
        self.pages_and_chunks_Pdf = []
        self.first_chapter_page = 0
        self.verbose = verbose
        self.embeddings = None
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)

    # Web Link and PDF Pipeline Setup
    def run_web_pipeline(self):
        print("🔗 Running Web Pipeline...")
        #----------------------------------------- Weblink PipeLine ----------------------------------------------------

        #________________Getting URL___________________________
        self.url_list = self.get_google_results(self.topic, self.number_results)
        print(self.url_list)

        #________________Scrapping Webpages from URL___________________________
        # The object self.scraped_documents is updated with the scraped and clean pages as result
        self.scrape_and_clean_pages()
        # print(self.scraped_documents)

        # Pick a random document to check for results
        random_doc = random.choice(self.scraped_documents)

        # Print sample Sample and total documents in the results
        # print(f"🔢 Total documents: {len(self.scraped_documents)}")
        # print("📄 Title:", random_doc.metadata.get("title", "No Title"))
        # print("📝 Content Sample:\n", random_doc.page_content[:300], "...")
        # df = pd.DataFrame(self.scraped_documents)
        # print(df.head())

        # ________________Chunking Web Links Data___________________________
        # The object self.pages_and_chunks_WebLinks is updated with the ChunkingLinkData
        self.ChunkingLinksData()
        # Pick random key value pair to check the results of chunking Picking a random dictionary
        # random_item = random.choice(self.pages_and_chunks_WebLinks)
        # # Print all key-value pairs from a random selected sample
        # for key, value in random_item.items():
        #     if key == "metadata" and isinstance(value, dict):
        #         print("\n📁 metadata:")
        #         for meta_key, meta_value in value.items():
        #             print(f"  {meta_key}: {meta_value}")
        #     else:
        #         print(f"\n🔑 {key}:\n{value}")

        # ________________Embedding Chunks of Web Pages___________________________
        self.embed_chunks()

    def print_wrapped(self, text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        print(wrapped_text)

    def Semantic_RAG_Search(self, query, rag_search_type):
        """
        Unified interface to run any RAG search method dynamically.
        """
        data = None

        # Attempt to load the embeddings from the .pkl file
        try:
            with open(self.save_path_Weblinks, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: File '{self.save_path_Weblinks}' not found. Skipping semantic search.")
            return [], "No Results - Embedding file missing"

        # Validate loaded data
        if not data or "chunks" not in data or "embeddings" not in data:
            print("❌ Invalid or incomplete data in the pickle file.")
            return [], "No Results - Incomplete Embedding Data"

        chunks = data["chunks"]
        embeddings = data["embeddings"]

        text_chunks_and_embedding_df = pd.DataFrame(chunks)
        text_chunks_and_embedding_df["embedding"] = list(embeddings)

        self.pages_and_chunks_WebLinks = text_chunks_and_embedding_df.to_dict(orient="records")
        self.embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()),
                                       dtype=torch.float32).to(self.device)

        print("✅ Loaded embedded data:")
        print(self.embeddings.shape)
        print(text_chunks_and_embedding_df.head())

        # 🔍 Dynamic search method map
        search_methods = {
            "dot product": (self.SearchQueryFromPickle_DOTPRODUCT, "Dot Product Search"),
            "cosine": (self.SearchQueryFromPickle_COSINE, "Cosine Similarity Search"),
            "euclidean": (self.SearchQueryFromPickle_EUCLIDEAN, "Euclidean Distance Search"),
            "faiss": (self.SearchQueryFromPickle_FAISS, "FAISS (IVF, HNSW, Flat) Search"),
            "hybrid": (self.SearchQueryFromPickle_HYBRID, "Hybrid Search (BM25 + Embeddings)"),
            "hybrid_bm25_embeddings": (self.SearchQueryFromPickle_HYBRID, "Hybrid Search (BM25 + Embeddings)"),
            "ann": (self.SearchQueryFromPickle_ANN, "Approximate Nearest Neighbors Search"),
            "cross_encoder": (self.SearchQueryWithCrossEncoder_Reranking, "Cross-Encoder Reranking")
        }

        # Normalize input for matching
        rag_search_type_clean = rag_search_type.strip().lower()

        search_fn, search_method = search_methods.get(
            rag_search_type_clean,
            (self.SearchQueryFromPickle_DOTPRODUCT, "Dot Product Search (Default)")
        )

        results = search_fn(query, self.pages_and_chunks_WebLinks, self.embeddings)
        return results, search_method

    def embed_chunks(self, ):

        def hash_chunk_text(text):
            return hashlib.md5(text.encode("utf-8")).hexdigest()

        if not self.pages_and_chunks_WebLinks:
            print("⚠️ No chunks available to embed.")
            return

        current_chunk_map = {
            hash_chunk_text(chunk["sentence_chunk"]): chunk for chunk in self.pages_and_chunks_WebLinks
        }

        if os.path.exists(self.save_path_Weblinks):
            print(f"📂 Loading existing embeddings from {self.save_path_Weblinks}")
            with open(self.save_path_Weblinks, "rb") as f:
                data = pickle.load(f)
                existing_embeddings = torch.tensor(data["embeddings"], device=self.device)
                existing_chunks = data["chunks"]
                existing_hashes = {hash_chunk_text(chunk["sentence_chunk"]) for chunk in existing_chunks}

            new_hashes = set(current_chunk_map.keys()) - existing_hashes
            new_chunks = [current_chunk_map[h] for h in new_hashes]

            if not new_chunks:
                print("✅ All chunks already embedded. No new chunks to add.")
                print(f"📊 Total existing chunks: {len(existing_chunks)}")
                return

            print(f"➕ New chunks found: {len(new_chunks)}")
            texts_to_embed = [chunk["sentence_chunk"] for chunk in new_chunks]
            new_embeddings = self.embedding_model.encode(
                texts_to_embed, show_progress_bar=True, convert_to_tensor=True
            )

            combined_embeddings = torch.cat([
                existing_embeddings,
                new_embeddings.to(self.device)
            ], dim=0)
            combined_chunks = existing_chunks + new_chunks

            with open(self.save_path_Weblinks, "wb") as f:
                pickle.dump({
                    "embeddings": combined_embeddings.cpu().numpy(),
                    "chunks": combined_chunks
                }, f)

            print("✅ Updated embeddings saved.")
            print(f"📊 Previous: {len(existing_chunks)} | Added: {len(new_chunks)} | Total: {len(combined_chunks)}")

        else:
            print("📁 No existing embeddings found. Creating new embedding file...")
            texts = [chunk["sentence_chunk"] for chunk in self.pages_and_chunks_WebLinks]
            self.embeddings = self.embedding_model.encode(
                texts, show_progress_bar=True, convert_to_tensor=True
            )

            with open(self.save_path_Weblinks, "wb") as f:
                pickle.dump({
                    "embeddings": self.embeddings.cpu().numpy(),
                    "chunks": self.pages_and_chunks_WebLinks
                }, f)

            print("✅ Embeddings generated and saved.")
            print(f"📊 Total new chunks embedded: {len(self.pages_and_chunks_WebLinks)}")

    def ChunkingLinksData(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1280,  # 320 tokens × 4 characters per token
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )


        for doc in tqdm(self.scraped_documents):
            splits = text_splitter.split_text(doc.page_content)
            for i, split in enumerate(splits):
                self.pages_and_chunks_WebLinks.append({
                    "sentence_chunk": split.strip(),
                    "chunk_index": i,
                    "chunk_word_count": len(split.split()),
                    "chunk_char_count": len(split),
                    "chunk_token_count": len(split) / 4,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "N/A")
                })

        print(f"✅ Total chunks generated: {len(self.pages_and_chunks_WebLinks)}")
        print(self.pages_and_chunks_WebLinks)


    def get_google_results(self, query: str, num_results: int) -> list[str]:
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


    def scrape_and_clean_pages(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        text_dir = Path("scraped_texts")
        text_dir.mkdir(exist_ok=True)

        seen_texts = set()

        # Initialize sentencizer
        nlp = English()
        nlp.add_pipe("sentencizer")

        def clean_text(text: str) -> str:
            text = re.sub(r'\s+', ' ', text)
            text = text.encode('ascii', errors='ignore').decode()
            text = re.sub(r'\.([A-Z])', r'. \1', text)
            boilerplate_keywords = [
                'cookie policy', 'terms of service', 'privacy policy',
                'enable javascript', 'ad blocker', 'copyright',
                'all rights reserved'
            ]
            for phrase in boilerplate_keywords:
                text = re.sub(rf'\b{re.escape(phrase)}\b', '', text, flags=re.IGNORECASE)
            lines = text.split('. ')
            if len(lines) > 10:
                most_common = max(set(lines), key=lines.count)
                if lines.count(most_common) > 3:
                    text = '. '.join([line for line in lines if line != most_common])
            return text.strip()

        def clean_html_with_readability_and_html_removal(html_content):
            doc = ReadabilityDocument(html_content)
            title = doc.short_title()
            summary_html = doc.summary()
            soup = BeautifulSoup(summary_html, 'html.parser')
            for tag in soup(
                    ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'table', 'ul', 'ol',
                     'dl']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            cleaned_text = re.sub(r'\s+', ' ', html.unescape(text)).strip()
            return cleaned_text, title

        def fetch_and_cache(url):
            url_hash = hashlib.md5(url.encode()).hexdigest()
            cache_path = cache_dir / f"{url_hash}.json"
            text_path = text_dir / f"{url_hash}.txt"

            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return Document(page_content=data["text"], metadata={
                        "source": data["url"],
                        "title": data.get("title", ""),
                        "text_path": data.get("text_path", "")
                    })

            try:
                response = requests.get(url, timeout=10)
                raw_text, title = clean_html_with_readability_and_html_removal(response.text)

                if not raw_text or len(raw_text) < 100:
                    print(f"⚠️ Skipping low-content page: {url}")
                    return None
                if any(bad_phrase in raw_text.lower() for bad_phrase in [
                    "please enable js", "ad blocker", "enable javascript", "bot detection"
                ]):
                    print(f"⚠️ Skipping bot-protected or misleading page: {url}")
                    return None

                final_text = clean_text(raw_text)

                try:
                    if detect(final_text) != "en":
                        print(f"🌐 Skipping non-English page: {url}")
                        return None
                except Exception as lang_err:
                    print(f"⚠️ Language detection failed: {lang_err}")
                    return None

                hash_digest = hashlib.md5(final_text.encode()).hexdigest()
                if hash_digest in seen_texts:
                    print(f"⚠️ Skipping duplicate content: {url}")
                    return None
                seen_texts.add(hash_digest)

                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(final_text)

                # 🧠 Sentence splitting here
                doc_nlp = nlp(final_text)
                sentences = [str(sent).strip() for sent in doc_nlp.sents if sent.text.strip()]
                sentence_count = len(sentences)

                data = {
                    "url": url,
                    "title": title,
                    "text": final_text,
                    "text_path": str(text_path)
                }

                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f)

                return Document(
                    page_content=final_text,
                    metadata={
                        "source": url,
                        "title": title,
                        "text_path": str(text_path),
                        "sentences": sentences,
                        "sentence_count": sentence_count
                    })

            except Exception as e:
                print(f"❌ Failed to scrape {url}: {e}")
                return None

        documents = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_and_cache, url): url for url in self.url_list}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    documents.append(result)

        self.scraped_documents = documents
        return documents


    def is_likely_toc_page(self, text):
        toc_keywords = ["contents", "table of contents", "index"]
        page_numbers = re.findall(r"\b\d{1,3}\b", text)
        dot_leaders = re.findall(r"\.{5,}", text)

        return (
            any(kw in text.lower() for kw in toc_keywords)
            or len(page_numbers) > 15
            or len(dot_leaders) >= 5
        )

    def find_first_chapter_page_auto_skip(self):
        scan_limit = 10
        chapter_patterns = [
            r"^\s*chapter\s+one\s*$",
            r"^\s*chapter\s+1\s*$",
            r"^\s*chapter\s+[0-9ivxlc]+\s*$",
            r"^\s*1[\.\s]+[a-zA-Z]",
            r"^\s*chapter\s+[a-z]+",
        ]
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in chapter_patterns]

        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")

        skip_pages = set()
        for page_number in range(min(scan_limit, len(doc))):
            text = doc[page_number].get_text()
            if self.is_likely_toc_page(text):
                skip_pages.add(page_number)

        for page_number, page in enumerate(doc):
            if page_number in skip_pages:
                continue
            text = page.get_text()
            lines = text.split("\n")
            for line in lines:
                for pattern in compiled_patterns:
                    if pattern.match(line.strip()):
                        return page_number + 1
        return 1

    def read_pdf_pages(self):
        doc = fitz.open(stream=self.pdf_bytes, filetype="pdf")
        pages_and_text = []
        for page_number, page in enumerate(doc):
            text = page.get_text()
            pages_and_text.append({
                "page_number": page_number,
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split())
            })
        doc.close()
        return pages_and_text

    def display_pdf_pages(self, st_container):
        if not self.pages_and_text_list:
            st_container.warning("No PDF data to display.")
            return

        st_container.subheader("📄 Extracted PDF Pages")
        for page in self.pages_and_text_list:
            st_container.markdown(f"### 📃 Page {page['page_number'] + 1}")
            st_container.text_area(
                label="",
                value=page['text'],
                height=300,
                key=f"page_text_{page['page_number']}"
            )

    # Search Functions Different Methods
    def SearchQueryFromPickle_DOTPRODUCT(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5):
        """
        Runs a query over provided or saved embedded data using dot product similarity.
        Returns top_k most similar chunks.
        """
        results = []
        try:
            # Step 1: Load data from pickle if needed
            if pages_and_chunks is None or embeddings_tensor is None:
                with open("EmbeddingStorage/EmbeddedData.pkl", "rb") as f:
                    data = pickle.load(f)

                chunks = data["chunks"]
                embeddings = data["embeddings"]

                df = pd.DataFrame(chunks)
                df["embedding"] = list(embeddings)

                pages_and_chunks = df.to_dict(orient="records")
                embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32)

            # Step 2: Make sure embeddings_tensor is on correct device
            embeddings_tensor = embeddings_tensor.to(self.device)

            # Step 3: Embed the query and move to device
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.to(self.device)
            else:
                raise TypeError("❌ query_embedding is not a torch.Tensor!")

            # Step 4: Compute dot product scores
            dot_scores = util.dot_score(query_embedding, embeddings_tensor)[0]
            top_results = torch.topk(dot_scores, k=top_k)

            print(f"\n📌 Query: {query}")
            print("🔍 Top results via Dot Product:\n")

            for score, idx in zip(top_results[0], top_results[1]):
                result = {
                    "score": float(score),
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during dot product search: {e}")

        return results

    def SearchQueryFromPickle_COSINE(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5):
        """Runs a query over provided or saved embedded data."""
        results = []
        try:
            # If not passed, load from pickle
            if pages_and_chunks is None or embeddings_tensor is None:
                with open("EmbeddingStorage/EmbeddedData.pkl", "rb") as f:
                    data = pickle.load(f)

                chunks = data["chunks"]
                embeddings = data["embeddings"]

                df = pd.DataFrame(chunks)
                df["embedding"] = list(embeddings)

                pages_and_chunks = df.to_dict(orient="records")
                embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(
                    self.device)
            # Embed the query
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)

            # Different Methods to Search Results

            # Dot product scores
            # dot_scores = util.dot_score(query_embedding, embeddings_tensor)[0]
            # top_results = torch.topk(dot_scores, k=top_k)

            # Cosine Product Scores
            # cosine_scores = util.cos_sim(query_embedding, embeddings_tensor)[0]
            cosine_scores = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), embeddings_tensor,
                                                                  dim=1)

            top_results = torch.topk(cosine_scores, k=top_k)

            print(f"\n📌 Query: {query}")
            print("🔍 Top results:\n")

            for score, idx in zip(top_results[0], top_results[1]):
                result = {
                    "score": float(score),
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during search from pickle: {e}")

        return results

    def SearchQueryFromPickle_FAISS(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5):
        """Runs a query over provided or saved embedded data using FAISS for similarity search."""
        results = []

        try:
            # Step 1: Load data if not passed
            if pages_and_chunks is None or embeddings_tensor is None:
                with open("EmbeddingStorage/EmbeddedData.pkl", "rb") as f:
                    data = pickle.load(f)

                chunks = data["chunks"]
                embeddings = data["embeddings"]

                df = pd.DataFrame(chunks)
                df["embedding"] = list(embeddings)

                pages_and_chunks = df.to_dict(orient="records")
                embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(
                    self.device)

            # Step 2: Embed the query
            query_embedding = self.embedding_model.encode(query)
            query_vector = query_embedding.reshape(1, -1).astype("float32")

            # Step 3: Convert corpus embeddings to numpy float32
            embeddings_np = embeddings_tensor.cpu().numpy().astype("float32")

            # Step 4: Build FAISS index
            dim = embeddings_np.shape[1]
            index = faiss.IndexFlatIP(dim)  # or IndexFlatL2 for Euclidean
            index.add(embeddings_np)

            # Step 5: Search using FAISS
            scores, indices = index.search(query_vector, top_k)

            print(f"\n📌 Query: {query}")
            print("🔍 Top results via FAISS:\n")

            for i in range(top_k):
                idx = indices[0][i]
                result = {
                    "score": float(scores[0][i]),
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during FAISS search from pickle: {e}")

        return results

    def SearchQueryFromPickle_HYBRID(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5, alpha=0.5):
        """
        Hybrid Search using BM25 lexical score + embedding similarity (cosine).
        alpha = weight for embedding score (0 to 1), 1 means pure embedding.
        """
        results = []

        try:
            # Step 1: Load data if not passed
            if pages_and_chunks is None or embeddings_tensor is None:
                with open("EmbeddingStorage/EmbeddedData.pkl", "rb") as f:
                    data = pickle.load(f)

                chunks = data["chunks"]
                embeddings = data["embeddings"]

                df = pd.DataFrame(chunks)
                df["embedding"] = list(embeddings)

                pages_and_chunks = df.to_dict(orient="records")
                embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(
                    self.device)

            # Step 2: Build BM25 Corpus
            corpus = [chunk["sentence_chunk"].split() for chunk in pages_and_chunks]
            bm25 = BM25Okapi(corpus)

            # Step 3: BM25 Scores
            bm25_scores = bm25.get_scores(query.split())

            # Step 4: Embedding Similarity Scores (Cosine)
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
            cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, embeddings_tensor)[0]
            embedding_scores = cosine_scores.cpu().numpy()

            # Step 5: Combine Scores
            final_scores = alpha * embedding_scores + (1 - alpha) * bm25_scores

            # Step 6: Get Top Results
            top_indices = np.argsort(final_scores)[::-1][:top_k]

            print(f"\n📌 Query: {query}")
            print("🔍 Top results via HYBRID BM25 + Embeddings:\n")

            for idx in top_indices:
                result = {
                    "score": float(final_scores[idx]),
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during HYBRID search: {e}")

        return results

    def SearchQueryFromPickle_EUCLIDEAN(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5):
        """
        Runs a query over embedded data using Euclidean Distance (L2).
        Returns top_k closest chunks (smallest distance = most similar).
        """
        results = []

        try:
            # Step 1: Load data if not passed
            if pages_and_chunks is None or embeddings_tensor is None:
                with open("EmbeddingStorage/EmbeddedData.pkl", "rb") as f:
                    data = pickle.load(f)

                chunks = data["chunks"]
                embeddings = data["embeddings"]

                df = pd.DataFrame(chunks)
                df["embedding"] = list(embeddings)

                pages_and_chunks = df.to_dict(orient="records")
                embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(
                    self.device)

            # Step 2: Encode query
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)

            # Step 3: Compute Euclidean distances
            diff = embeddings_tensor - query_embedding
            distances = torch.norm(diff, dim=1)  # L2 norm (Euclidean distance)

            # Step 4: Get top-k closest (smallest distances)
            top_indices = torch.topk(-distances, k=top_k).indices.cpu().numpy()

            print(f"\n📌 Query: {query}")
            print("🔍 Top results via Euclidean Distance:\n")

            for idx in top_indices:
                result = {
                    "score": float(distances[idx]),  # smaller = better
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during Euclidean search: {e}")

        return results

    def SearchQueryFromPickle_ANN(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5):
        """
        Runs a query over embedded data using Approximate Nearest Neighbors (via FAISS).
        Returns top_k most similar chunks.
        """
        results = []

        try:
            # Step 1: Load data if not passed
            if pages_and_chunks is None or embeddings_tensor is None:
                with open("EmbeddingStorage/EmbeddedData.pkl", "rb") as f:
                    data = pickle.load(f)

                chunks = data["chunks"]
                embeddings = data["embeddings"]

                df = pd.DataFrame(chunks)
                df["embedding"] = list(embeddings)

                pages_and_chunks = df.to_dict(orient="records")
                embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(
                    self.device)

            # Step 2: Encode the query
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).to(self.device)
            query_np = query_embedding.cpu().numpy().astype("float32")

            # Step 3: Prepare FAISS index
            embeddings_np = embeddings_tensor.cpu().numpy().astype("float32")
            index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 (Euclidean) index

            if not index.is_trained:
                index.train(embeddings_np)

            index.add(embeddings_np)

            # Step 4: Search
            distances, indices = index.search(query_np.reshape(1, -1), top_k)

            print(f"\n📌 Query: {query}")
            print("🔍 Top results via ANN (FAISS - L2):\n")

            for idx, dist in zip(indices[0], distances[0]):
                result = {
                    "score": float(dist),  # smaller = better
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during ANN search: {e}")

        return results

    def SearchQueryWithCrossEncoder_Reranking(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5):
        """
        Uses a Cross-Encoder model to rerank chunks based on semantic similarity to the query.
        Assumes candidate chunks are first retrieved using any base vector search method.
        Returns top_k most relevant chunks.
        """
        results = []

        try:
            # Step 1: Load data if not passed
            if pages_and_chunks is None or embeddings_tensor is None:
                with open("EmbeddingStorage/EmbeddedData.pkl", "rb") as f:
                    data = pickle.load(f)

                chunks = data["chunks"]
                embeddings = data["embeddings"]

                df = pd.DataFrame(chunks)
                df["embedding"] = list(embeddings)

                pages_and_chunks = df.to_dict(orient="records")
                embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(
                    self.device)

            # Step 2: Retrieve top N candidates using fast base method
            base_top_k = 15  # Get more than final top_k for reranking
            if hasattr(self, 'SearchQueryFromPickle_DOTPRODUCT'):
                base_results = self.SearchQueryFromPickle_DOTPRODUCT(query, pages_and_chunks, embeddings_tensor,
                                                                     top_k=base_top_k)
            else:
                raise Exception("No base search method (e.g., DOTPRODUCT) defined.")

            # Step 3: Load Cross-Encoder model
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device)

            # Step 4: Rerank using Cross-Encoder scores
            pairs = [[query, item["text"]] for item in base_results]
            scores = cross_encoder.predict(pairs)

            # Step 5: Sort and return top-k based on rerank scores
            reranked = sorted(zip(base_results, scores), key=lambda x: x[1], reverse=True)[:top_k]

            print(f"\n📌 Query: {query}")
            print("🔍 Top results via Cross-Encoder Reranking:\n")

            for item, score in reranked:
                result = {
                    "score": float(score),
                    "text": item["text"],
                    "source": item.get("source", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during Cross-Encoder reranking: {e}")

        return results
