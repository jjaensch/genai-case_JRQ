import os
DEFAULT_OUTPUT_DIR = r"../outputs/"

class MarkdownWriter:
    
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.abspath(os.path.join(base_dir, DEFAULT_OUTPUT_DIR))

    def write_md(self, output_data: str, file_name: str) -> None:
        enhanced_file_name = file_name.replace('.md', '_enhanced.md')
        file_path = os.path.join(self.output_dir, enhanced_file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output_data)