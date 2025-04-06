import os

DATA_DIR = r"../data/"
PROMPT_DIR = r"../prompt/"

DATA_FILES = os.listdir(DATA_DIR)
PROMPT_FILES = os.listdir(PROMPT_DIR)

#%%
if len(PROMPT_FILES) == 1 and PROMPT_FILES[0].endswith('.md'):
    file_path = os.path.join(PROMPT_DIR, PROMPT_FILES[0])
    with open(file_path, 'r', encoding='utf-8') as file:
        PROMPT_TEXT = file.read()
else:
    raise ValueError("There should be only one prompt file in the prompt directory, and prompt format msut be .md.")

#%%
input_data_strs = []
for file in DATA_FILES:
    if file.endswith(".md"):
        file_path = os.path.join(DATA_DIR, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            input_data_strs.append(f.read())
    else:
        raise ValueError(f"File {file} is not in .md format. Please check the file format. Reading skipped.")
