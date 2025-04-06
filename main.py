from src.readers import MarkdownLoader
from src.danskGPT_tiny import DanskGPTTiny

def main():
    
    # Prepare model
    model = DanskGPTTiny()

    # prepare input and prompt
    text_loader = MarkdownLoader()
    project_descriptions = text_loader.load_input_data()
    prompt_text = text_loader.load_prompt()

    # - w data, prompt model
    for description in project_descriptions:
        output = model.run_prompt_single_input(description, prompt_text)
        # Output enhancements
        print(output)

    ## Printouts for test
    # print("!#!#!#!#!#!#!#Prompt Text!#!#!#!#!#!#!#")
    # print(prompt_text, "\n")
    # for description in project_descriptions:
    #     print(f"!#!#!#!#!#!#!#Project Description {project_descriptions.index(description) + 1} of {len(project_descriptions)}!#!#!#!#!#!#!#")
    #     print(description)
    #     print("\n")
    # print("End of Script\n")

if __name__ == "__main__":
    main()
