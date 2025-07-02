from ImportsForModel import *  # Assumes fitz, re, nltk, spacy, pandas, tqdm, streamlit, etc.


class RAG_PDF_Application:
    def __init__(self, topic, number_results, mode, pdf_bytes=None, verbose=False, filename=None):
        os.makedirs("EmbeddingStorage", exist_ok=True)
        self.save_path_pdfData = "EmbeddingStorage/PDF_EmbeddedData.pkl"
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
        self.file_name = None
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)

        # Web Link and PDF Pipeline Setup
        # self.run_pdf_pipeline()

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

    def run_pdf_pipeline(self):
        # PDF Setup
        self.first_chapter_page = self.find_first_chapter_page_auto_skip()
        self.pages_and_text_list_pdf = self.read_pdf_pages()
        print(f"📍 First chapter starts at page: {self.first_chapter_page}")

        # Sentencizing the Data
        avg_sent_pdf, self.pages_and_text_list_pdf = self.Sentencizing_NLP(
            self.pages_and_text_list_pdf, self.file_name)

        # Chunking
        _, self.pages_and_chunks_pdf = self.Chunking_NLP(
            self.pages_and_text_list_pdf, avg_sent_pdf, source_type="pdf")

        # Splitting Chunks
        # Splitting each chunk into its own item
        self.pages_and_chunks_pdf = self.Split_Chunks(
            self.pages_and_text_list_pdf, source_type="pdf", min_token_length=30
        )
        print(self.pages_and_text_list_pdf)

        random_item = random.choice(self.pages_and_text_list_pdf)
        for key, value in random_item.items():
            if key == "metadata" and isinstance(value, dict):
                print("\n📁 metadata:")
                for meta_key, meta_value in value.items():
                    print(f"  {meta_key}: {meta_value}")
            else:
                print(f"\n🔑 {key}:\n{value}")

        # Run once then comment it out so that we get embeddings saved to a csv file
        # For PDF
        self.embed_chunks_universal(
            save_path=self.save_path_pdfData,
            pages_and_chunks=self.pages_and_chunks_pdf,
            source_type="pdf"
        )


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

    #******************************************************************************************************************

    def print_wrapped(self, text, wrap_length=80):
        wrapped_text = textwrap.fill(text, wrap_length)
        # print(wrapped_text)

    def Semantic_Rag_DotProduct_Search(self, query, rag_search_type):
        with open(self.save_path_pdfData, "rb") as f:
            data = pickle.load(f)

        chunks = data["chunks"]
        embeddings = data["embeddings"]

        text_chunks_and_embedding_df = pd.DataFrame(chunks)
        text_chunks_and_embedding_df["embedding"] = list(embeddings)

        self.pages_and_chunks_WebLinks = text_chunks_and_embedding_df.to_dict(orient="records")
        self.embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()),
                                       dtype=torch.float32).to(self.device)

        print("Loaded embedded data:")
        # print(self.embeddings.shape)
        # print(text_chunks_and_embedding_df.head())
        if rag_search_type == "dot product":
            results = self.SearchQueryFromPickle_DOTPRODUCT(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "Dot Product Search"

        elif rag_search_type == "cosine":
            results = self.SearchQueryFromPickle_COSINE(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "Cosine Similarity Search"

        elif rag_search_type == "euclidean":
            results = self.SearchQueryFromPickle_EUCLIDEAN(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "Euclidean Distance Search"

        elif rag_search_type == "faiss":
            results = self.SearchQueryFromPickle_FAISS(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "FAISS (IVF, HNSW, Flat) Search"

        elif rag_search_type == "hybrid_BM25_Embeddings":
            results = self.SearchQueryFromPickle_HYBRID(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "Hybrid Search (BM25 + Embeddings)"

        elif rag_search_type == "ann":
            results = self.SearchQueryFromPickle_ANN(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "(Approximate Nearest Neighbors) Search"

        elif rag_search_type == "cross_encoder":
            results = self.SearchQueryWithCrossEncoder_Reranking(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "Cross-Encoder Reranking"

        else:
            results = self.SearchQueryFromPickle_DOTPRODUCT(query, self.pages_and_chunks_WebLinks, self.embeddings)
            search_method = "Dot Product Search (Default)"

        for item in results:
            item["source"] = f"Page Number {item.get('page_number', 'Unknown')}"

        return results, search_method


    #---------------------------------------------------------------------------------------------------------------

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
            print(
                "***************************************************************************************************")

            for score, idx in zip(top_results[0], top_results[1]):
                result = {
                    "score": float(score),
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A"),
                    "page_number": pages_and_chunks[idx].get("page_number", "N/A")
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
                    "source": pages_and_chunks[idx].get("source", "N/A"),
                    "page_number": pages_and_chunks[idx].get("page_number", "N/A")
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
            print(
                "***************************************************************************************************")

            for i in range(top_k):
                idx = indices[0][i]
                result = {
                    "score": float(scores[0][i]),
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A"),
                    "page_number": pages_and_chunks[idx].get("page_number", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during FAISS search from pickle: {e}")

        return results

    def SearchQueryFromPickle_HYBRID(self, query, pages_and_chunks=None, embeddings_tensor=None, top_k=5,
                                     alpha=0.5):
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
            print(
                "***************************************************************************************************")

            for idx in top_indices:
                result = {
                    "score": float(final_scores[idx]),
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A"),
                    "page_number": pages_and_chunks[idx].get("page_number", "N/A")
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
            print(
                "***************************************************************************************************")

            for idx in top_indices:
                result = {
                    "score": float(distances[idx]),  # smaller = better
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A"),
                    "page_number": pages_and_chunks[idx].get("page_number", "N/A")
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
            print(
                "***************************************************************************************************")

            for idx, dist in zip(indices[0], distances[0]):
                result = {
                    "score": float(dist),  # smaller = better
                    "text": pages_and_chunks[idx]["sentence_chunk"],
                    "source": pages_and_chunks[idx].get("source", "N/A"),
                    "page_number": pages_and_chunks[idx].get("page_number", "N/A")
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
            base_top_k = 15  # Retrieve more than top_k to rerank
            if hasattr(self, 'SearchQueryFromPickle_DOTPRODUCT'):
                base_results = self.SearchQueryFromPickle_DOTPRODUCT(
                    query, pages_and_chunks, embeddings_tensor, top_k=base_top_k
                )
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
            print("***************************************************************************************************")

            for item, score in reranked:
                result = {
                    "score": float(score),
                    "text": item["text"],
                    "source": item.get("source", "N/A"),
                    "page_number": item.get("page_number", "N/A")
                }
                results.append(result)

        except Exception as e:
            print(f"❌ Error during Cross-Encoder reranking: {e}")

        return results




