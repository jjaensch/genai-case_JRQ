# Code based on documentation for danskGPT-tiny
# https://huggingface.co/mhenrichsen/danskgpt-tiny-chat 
from transformers import AutoModelForCausalLM, AutoTokenizer

class DanskGPTTiny:
    def __init__(self, 
                 MODEL_NAME='mhenrichsen/danskgpt-tiny-chat'):
        # Load the tokenizer and model from the transformers library
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def run_prompt_single_input(self, 
                                 input_str: str, prompt: str, 
                                 var_temperature: float = 0.8, var_top_p: float = 0.95, var_max_tokens: int = 1024):
        # Combine the prompt and input string
        new_prompt = f"{prompt}{input_str}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize the input prompt
        input_ids = self.tokenizer.encode(new_prompt, return_tensors="pt")

        # Generate output using the model
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=var_max_tokens,
            temperature=var_temperature,
            top_p=var_top_p,
            do_sample=True
        )

        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text