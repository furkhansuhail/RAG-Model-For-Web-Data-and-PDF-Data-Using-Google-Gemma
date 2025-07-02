from ImportsForModel import *  # Assumes fitz, re, nltk, spacy, pandas, tqdm, streamlit, etc.



if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []
# Hybrid RAG system: scraping, preprocessing, chunking, embedding, querying, and visualizing results in Streamlit.


class LLM_Application:
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
        # LLM
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
        # self.quantization_config = quantization_config
        # self.use_quantization_config = True



    def SearchModuleSetup_LLM(self):
        results  = self.LLM_Model_Setup(pdf_path=self.save_path_pdf_Embeddings,
                                               web_path=self.save_path_Weblinks_Embeddings,
                                               query= self.topic)
        return results, "LLM"

    def quantization_configuration_setup(self):
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = round(gpu_memory_bytes / (2 ** 30))
        print(f"Available GPU memory: {gpu_memory_gb} GB")
        use_quantization_config = None
        # Note: the following is Gemma focused, however, there are more and more LLMs of the 2B and 7B size appearing for local use.
        if gpu_memory_gb < 5.1:
            print(
                f"Your available GPU memory is {gpu_memory_gb}GB, you may not have enough memory to run a Gemma LLM locally without quantization.")
        elif gpu_memory_gb < 8.1:
            print(f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in 4-bit precision.")
            use_quantization_config = True
            model_id = "google/gemma-2b-it"
        elif gpu_memory_gb < 19.0:
            print(
                f"GPU memory: {gpu_memory_gb} | Recommended model: Gemma 2B in float16 or Gemma 7B in 4-bit precision.")
            use_quantization_config = False
            model_id = "google/gemma-2b-it"
        elif gpu_memory_gb > 19.0:
            print(f"GPU memory: {gpu_memory_gb} | Recommend model: Gemma 7B in 4-bit or float16 precision.")
            use_quantization_config = False
            model_id = "google/gemma-7b-it"

        print(f"use_quantization_config set to: {use_quantization_config}")
        print(f"model_id set to: {self.model_id}")

        return use_quantization_config

    def get_model_num_params(self, model: torch.nn.Module):
        return sum([param.numel() for param in model.parameters()])

    def get_model_mem_size(self, model: torch.nn.Module):
        """
        Get how much memory a PyTorch model takes up.

        See: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822
        """
        # Get model parameters and buffer sizes
        mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
        mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

        # Calculate various model sizes
        model_mem_bytes = mem_params + mem_buffers  # in bytes
        model_mem_mb = model_mem_bytes / (1024 ** 2)  # in megabytes
        model_mem_gb = model_mem_bytes / (1024 ** 3)  # in gigabytes

        return {"model_mem_bytes": model_mem_bytes,
                "model_mem_mb": round(model_mem_mb, 2),
                "model_mem_gb": round(model_mem_gb, 2)}



    def LLM_Model_Setup(self, pdf_path = None, web_path= None, query= None, similarity_method="dot_product"):
        use_quantization_config = self.quantization_configuration_setup()
        print(f"use_quantization_config set to: {use_quantization_config}")
        # Delegate to the correct method if needed
        # AutoModelForCausalLM

        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
        print(f"[INFO] Using attention implementation: {attn_implementation}")

        # 2. Pick a model we'd like to use (this will depend on how much GPU memory you have available)
        # model_id = "google/gemma-7b-it"
        model_id = self.model_id  # (we already set this above)
        print(f"[INFO] Using model_id: {model_id}")

        # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

        # 4. Instantiate the model
        llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                         torch_dtype=torch.float16,  # datatype to use, we want float16
                                                         quantization_config=quantization_config if use_quantization_config else None,
                                                         low_cpu_mem_usage=False,  # use full memory
                                                         attn_implementation=attn_implementation)  # which attention version to use

        if not use_quantization_config:  # quantization takes care of device setting automatically, so if it's not used, send model to GPU
            llm_model.to("cuda")

        print(llm_model)
        print("_________________________________________________________________")
        print(self.model_id)
        print("________________________________________________________________")
        print(self.get_model_num_params(llm_model))
        print("________________________________________________________________")
        print(self.get_model_mem_size(llm_model))
        print("________________________________________________________________")

        input_text = self.topic
        print(f"Input text:\n{input_text}")

        # Create prompt template for instruction-tuned model
        dialogue_template = [
            {"role": "user",
             "content": input_text}
        ]

        # Apply the chat template
        prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                               tokenize=False,  # keep as raw text (not tokenized)
                                               add_generation_prompt=True)
        print(f"\nPrompt (formatted):\n{prompt}")

        # Tokenize the input text (turn it into numbers) and send it to GPU
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
        print(f"Model input (tokenized):\n{input_ids}\n")

        # Generate outputs passed on the tokenized input
        # See generate docs: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/text_generation#transformers.GenerationConfig
        outputs = llm_model.generate(**input_ids,
                                     max_new_tokens=2048)  # define the maximum number of new tokens to create
        # print(f"Model output (tokens):\n{outputs[0]}\n")

        # Decode the output tokens to text
        outputs_decoded = tokenizer.decode(outputs[0])
        print(f"Model output (decoded):\n{outputs_decoded}\n")
        return outputs_decoded












