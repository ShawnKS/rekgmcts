from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

class HUGGINGFACE_LLM:
    def __init__(self, local_model_path, max_new_tokens=256, n=1):
        """
        Initialize the HuggingFace LLM class.

        Parameters:
        local_model_path (str): Local model file path.
        n (int): Number of sequences to generate.
        """
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.pipe = self.create_pipe(local_model_path, n=n)
        self.answer_prompt = ''

    def create_pipe(self, local_model_path, n):
        """
        Create an instance of the HuggingFace pipeline.

        Parameters:
        local_model_path (str): Local model file path.
        n (int): Number of sequences to generate.

        Returns:
        pipe (pipeline): Text generation pipeline.
        """
        # Optional configurations (such as loading in 4bit, quantization, etc.), remove if not needed
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load tokenizer and model from local path
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)  # Load tokenizer from local path
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,                   # Load model from local path
            device_map="auto",                  # Automatically select device (recommended for GPU)
            trust_remote_code=True,             # If the model uses custom code logic
            # quantization_config=bnb_config      # Optional: Quantization configuration
        )

        # Load the pipeline
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,                    # Enable caching
            device_map="auto",                 # Automatically assign device
            max_new_tokens=self.max_new_tokens,                 # Maximum number of new tokens to generate
            # do_sample=True,                    # Enable random sampling
            do_sample=False,
            top_k=1,                           # Select top k most probable tokens
            num_return_sequences=n,            # Number of sequences to generate at once
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            trust_remote_code=True,             # If the model uses custom code logic
            return_full_text=False  # Return the complete text (including input)
        )

        return pipe
    
    def verify(self, answer, true_answer, question):
        prompt = format_evaluation_prompt(question, true_answer, answer)
        result = self(prompt)[0]
        value_match = re.search("\d\.\d+", result)
        if value_match is None:
            print(f"Could not get value of '{result}'")
            value = 0.0
        else:
            value = float(value_match[0])
        return value

    def generate_answer(self, question, cluster_chain_of_entities):
        prompt = self.answer_prompt + question + '\n'
        
        # Format triplets as required
        if len(cluster_chain_of_entities) == 0:
            # If there are no triplets, return an answer based on common knowledge
            prompt += "Knowledge Triplets: "
            raw_ans = self(prompt)[0]
            ans = extract_answer(raw_ans)
            return raw_ans, ans
        else:
            # Format triplets as required
            chain_prompt = ', '.join([f"{head} -> {relation} -> {tail}" for head, relation, tail in cluster_chain_of_entities])
            triples = f"Knowledge Triplets: {chain_prompt}\nA: "
            print(triples)
            prompt = prompt + triples
            print(prompt)
            # Call the model to generate an answer
            raw_ans = self(prompt)[0]
            ans = extract_answer(raw_ans)
            return raw_ans, ans

    def set_answer_prompt(self, prompt):
        self.answer_prompt = prompt

    def __call__(self, prompt, n=None, stop=None, temperature=0.5):
        n = n or self.n
        self.pipe.temperature = temperature  # Dynamically set temperature
        result = self.pipe(prompt)
        return [e['generated_text'] for e in result]

import openai

class GPT4_LLM:
    def __init__(self, api_key, max_tokens=40, n=1):
        """
        Initialize the GPT-4 API integration for text generation.

        Parameters:
        api_key (str): OpenAI API key.
        n (int): Number of sequences to generate.
        """
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.n = n
        openai.api_key = api_key
        self.answer_prompt = ''

    def generate_answer(self, question, cluster_chain_of_entities):
        prompt = self.answer_prompt + question + '\n'
        
        # Format triplets if available
        if len(cluster_chain_of_entities) == 0:
            prompt += "Knowledge Triplets: "
        else:
            chain_prompt = ', '.join([f"{head} -> {relation} -> {tail}" for head, relation, tail in cluster_chain_of_entities])
            prompt += f"Knowledge Triplets: {chain_prompt}\nA: "

        response = openai.Completion.create(
            model="gpt-4",   # Use GPT-4 model
            prompt=prompt,
            max_tokens=self.max_tokens,
            n=self.n,
            temperature=0.5,
            stop=None
        )

        # Process and return the result
        raw_ans = response['choices'][0]['text']
        ans = extract_answer(raw_ans)
        return raw_ans, ans

    def set_answer_prompt(self, prompt):
        self.answer_prompt = prompt