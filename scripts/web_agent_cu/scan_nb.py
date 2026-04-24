import json
nb = json.load(open("/tmp/cu.ipynb"))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    low = src.lower()
    if any(k in low for k in ["system", "prompt", "computer_use", "tool_call", "function", "assistant"]):
        print(f"===CELL {i} type={cell['cell_type']}===")
        print(src[:6000])
        print()
