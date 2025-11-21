#!/usr/bin/env python3
import subprocess
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    
    tex_files = list(script_dir.glob("*.tex"))
    
    for tex_file in tex_files:
        print(f"Processing {tex_file.name}...")
        
        try:
            result = subprocess.run(
                ["latexmk", "-pdf", "-output-directory=" + str(script_dir), str(tex_file)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Generated PDF for {tex_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to generate PDF for {tex_file.name}: {e}")
            continue
        
        pdf_file = script_dir / f"{tex_file.stem}.pdf"
        svg_file = script_dir / f"{tex_file.stem}.svg"
        
        if pdf_file.exists():
            try:
                result = subprocess.run(
                    ["pdf2svg", str(pdf_file), str(svg_file)],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Insert white background as the first child of the <svg> node
                BACKGROUND = '<rect width="100%" height="100%" fill="white"/>'
                
                # Read the SVG file and modify it
                with open(svg_file, 'r') as f:
                    svg_content = f.read()
                
                # Find the opening <svg> tag and insert background after it
                svg_start = svg_content.find('<svg')
                if svg_start != -1:
                    svg_tag_end = svg_content.find('>', svg_start)
                    if svg_tag_end != -1:
                        modified_content = (svg_content[:svg_tag_end + 1] + 
                                          '\n' + BACKGROUND + 
                                          svg_content[svg_tag_end + 1:])
                    
                    # Write back the modified content
                    with open(svg_file, 'w') as f:
                        f.write(modified_content)

                print(f"Generated SVG for {tex_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate SVG for {tex_file.name}: {e}")

if __name__ == "__main__":
    main()
