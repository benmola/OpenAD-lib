
import os
import json
import re

notebooks_dir = os.path.join(os.path.dirname(__file__), 'notebooks')

# Regex to find pip install commands with extras
# Matches: !pip install "git+url#egg=name[extra]" or without quotes
# We want to remove the [extra] part and ensure quotes

def fix_pip_line(line):
    if "pip install" not in line or "OpenAD-lib" not in line:
        return line
        
    # Standard target line
    new_line = '!pip install git+https://github.com/benmola/OpenAD-lib.git'
    
    # Check if commented
    if line.strip().startswith("#"):
        return "# " + new_line + "\n"
    else:
        return new_line + "\n"

def process_notebooks():
    count = 0
    for filename in os.listdir(notebooks_dir):
        if not filename.endswith(".ipynb"):
            continue
            
        filepath = os.path.join(notebooks_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            modified = False
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    new_source = []
                    for line in cell['source']:
                        if "pip install" in line and "OpenAD-lib" in line:
                            fixed_line = fix_pip_line(line)
                            if fixed_line != line:
                                new_source.append(fixed_line)
                                modified = True
                            else:
                                new_source.append(line)
                        else:
                            new_source.append(line)
                    cell['source'] = new_source
            
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=4)
                print(f"✅ Updated {filename}")
                count += 1
            else:
                print(f"⚪ No changes needed for {filename}")
                
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\nTotal notebooks updated: {count}")

if __name__ == "__main__":
    process_notebooks()
