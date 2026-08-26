import json

with open("split_audio.json", "r", encoding="utf-8") as f:
    split = json.load(f)

test_val = split["Accordion"]["test"]
print("Tip:", type(test_val))
print("Continut:", repr(test_val))