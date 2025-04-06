from src.readers import MarkdownLoader

def main():
    text_loader = MarkdownLoader()
    project_descriptions = text_loader.load_input_data()
    prompt_text = text_loader.load_prompt()
    
    print("!#!#!#!#!#!#!#Prompt Text!#!#!#!#!#!#!#")
    print(prompt_text, "\n")
    
    
    for description in project_descriptions:
        print(f"!#!#!#!#!#!#!#Project Description {project_descriptions.index(description) + 1} of {len(project_descriptions)}!#!#!#!#!#!#!#")
        print(description)
        print("\n")
    print("End of Script\n")

if __name__ == "__main__":
    main()
