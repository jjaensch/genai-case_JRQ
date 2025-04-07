from src.readers import MarkdownLoader
from src.danskGPT_tiny import LanguageModel
from src.writers import MarkdownWriter

def main():
    # Prepare model
    model = LanguageModel(MODEL_NAME='mhenrichsen/danskgpt-tiny-chat')
    # prepare .md input and prompt
    text_loader = MarkdownLoader()
    guiding_prompt = text_loader.load_prompt()
    project_descriptions_dict = text_loader.load_input_data()
    # - w data, prompt model
    text_writer = MarkdownWriter()
    for filename, project_desc in project_descriptions_dict.items():
        # Set variables for manual "parameter search"
        output = model.run_prompt_single_input(project_desc, 
                                               guiding_prompt,
                                               var_temperature = 0.2,
                                               )
        # - to file
        text_writer.write_md(output, filename)
        
if __name__ == "__main__":
    main()