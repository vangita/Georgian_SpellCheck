import json
import sys

def convert_notebook(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            code_cells.append(source)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n# Cell\n".join(code_cells))
    print(f"Converted {notebook_path} to {output_path}")

if __name__ == "__main__":
    convert_notebook(sys.argv[1], sys.argv[2])
