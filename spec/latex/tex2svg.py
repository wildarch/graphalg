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
                print(f"Generated SVG for {tex_file.name}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate SVG for {tex_file.name}: {e}")

if __name__ == "__main__":
    main()
