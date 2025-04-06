# Code based on documentation for danskGPT-tiny
# https://huggingface.co/mhenrichsen/danskgpt-tiny-chat 
from transformers import AutoModelForCausalLM, AutoTokenizer

class LanguageModel:
    def __init__(self, 
                 MODEL_NAME='mhenrichsen/danskgpt-tiny-chat'):
        # Load the tokenizer and model from the transformers library
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        # Håndtér manglende pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_prompt_single_input(self, 
                                 input_str: str, prompt: str, 
                                 var_temperature: float = 0.8, 
                                 var_top_p: float = 0.95, 
                                 var_max_tokens: int = 1024, #2048 is max for danskGPT-tiny
                                 var_repetition_penalty: float = 1.1,
                                 var_do_sample: bool = True,
                                 var_num_beams: int = 1,
                                 var_no_repeat_ngram_size: int = 2,
                                 var_length_penalty: float = 1.0
                                 ):

        prompt_trim = prompt.split('---')[0] #"Du er en hjælpsom assistent."
        new_prompt = (
            f"<|im_start|>system\n{prompt_trim.strip()}<|im_end|>\n"
            f"<|im_start|>user\n{input_str.strip()}<|im_end|>\n"
            "<|im_start|>assistant\n"
            )

        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(new_prompt.strip(), return_tensors="pt")

        if input_ids.shape[1] + var_max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Prompt + max_tokens overstiger modelens længdebegrænsning.")

        # Generate output using the model
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=var_max_tokens,
            temperature=var_temperature,
            top_p=var_top_p,
            repetition_penalty=var_repetition_penalty,
            do_sample=var_do_sample,
            num_beams=var_num_beams,
            no_repeat_ngram_size=var_no_repeat_ngram_size,
            length_penalty=var_length_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id  # <-- stop på korrekt sted
        )
        
        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], #remove input
                                                skip_special_tokens=True).strip()

        return generated_text