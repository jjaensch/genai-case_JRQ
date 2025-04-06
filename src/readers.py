import os

#%%
class MarkdownLoader:
    def __init__(self, 
                 DATA_DIR = r"../data/", 
                 PROMPT_DIR = r"../prompt/"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.abspath(os.path.join(base_dir, DATA_DIR))
        self.prompt_dir = os.path.abspath(os.path.join(base_dir, PROMPT_DIR))

    #%%
    def load_prompt(self) -> str:
        prompt_file_strs = os.listdir(self.prompt_dir)
        if len(prompt_file_strs) == 1 and prompt_file_strs[0].endswith('.md'):
            file_path = os.path.join(self.prompt_dir, prompt_file_strs[0])
            with open(file_path, 'r', encoding='utf-8') as file:
                prompt_text = file.read()
        else:
            raise ValueError("There should be only one prompt file in the prompt directory, and prompt format msut be .md.")
        return prompt_text

    #%%
    def load_input_data(self) -> dict[str, str]:
        data_file_strs = os.listdir(self.data_dir)
        input_data_dict = {}
        for file in data_file_strs:
            if file.endswith(".md"):
                file_path = os.path.join(self.data_dir, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    input_data_dict[file] = f.read()
            else:
                raise ValueError(f"File {file} is not in .md format. Please check the file format. Reading skipped.")
        return input_data_dict
