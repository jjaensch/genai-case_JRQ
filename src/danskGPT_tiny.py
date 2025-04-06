# Code based on documentation for danskGPT-tiny
# https://huggingface.co/mhenrichsen/danskgpt-tiny-chat 
from vllm import LLM, SamplingParams

class DanskGPTTiny:
    def __init__(self, 
                 MODEL_NAME = 'mhenrichsen/danskgpt-tiny-chat',):
        self.model = LLM(model=MODEL_NAME)

    def run_prompt_single_input (self, 
                                 input_str: str, prompt: str, 
                                 var_temperature: float = 0.8, var_top_p: float = 0.95, var_max_tokens: int = 1024
                                 ):
        # prepare model
        llm = self.model
        sampling_params = SamplingParams(var_temperature, var_top_p, var_max_tokens)

        # Initialize conversation history with system message
        ### system_message = "Du er en hjælpsom assistent."
        ### conversation_history = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n"
        
        new_prompt = f"{prompt}{input_str}<|im_end|>\n<|im_start|>assistant\n"
        output = llm.generate(new_prompt, sampling_params)
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     print(f"AI: {generated_text!r}")
        #     conversation_history = f"{prompt}{generated_text!r}<|im_end|>\n<|im_start|>user\n"
        return output