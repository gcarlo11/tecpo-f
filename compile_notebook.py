import json
import re

EMOJI_REPLACEMENTS = {
    "🏆": "[CHAMPION]",
    "⏱️": "[TIME]",
    "✅": "[OK]",
    "📁": "[DIR]",
    "⚠️": "[WARN]",
    "📊": "[DATA]",
    "🔍": "[AUDIT]",
    "📸": "[PLOT]",
    "ℹ️": "[INFO]",
    "🔧": "[FE]",
    "🚀": "[START]",
    "🎯": "[METRIC]",
    "📈": "[EVAL]",
    "🔥": "[ENSEMBLE]",
    "📦": "[ZIP]",
    "🎉": "[READY]"
}

with open(r"d:\vscode\tecpo-f\champion_pipeline.py", "r", encoding="utf-8") as f:
    text = f.read()

for emoji, tag in EMOJI_REPLACEMENTS.items():
    text = text.replace(emoji, tag)

with open(r"d:\vscode\tecpo-f\champion_pipeline.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Replaced emojis in champion_pipeline.py with ASCII tags.")

# Now parse champion_pipeline.py into Jupyter Notebook cells
raw_cells = text.split("# %%\n")
cells = []

for block in raw_cells:
    block = block.strip()
    if not block:
        continue
    
    if block.startswith("# %% [markdown]") or block.startswith("[markdown]"):
        lines = block.split("\n")
        # remove marker line
        md_lines = []
        for line in lines[1:]:
            # remove leading '# ' for markdown
            if line.startswith("# "):
                md_lines.append(line[2:] + "\n")
            elif line.startswith("#"):
                md_lines.append(line[1:] + "\n")
            else:
                md_lines.append(line + "\n")
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": md_lines
        })
    else:
        code_lines = [line + "\n" for line in block.split("\n")]
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_lines
        })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_out = r"d:\vscode\tecpo-f\tabular_mining_champion_template.ipynb"
with open(nb_out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook compiled with {len(cells)} cells at: {nb_out}")
