# test_cot.py
from utils import cot_generator as cg
import json

def main():
    q = "Does aspirin reduce risk of myocardial infarction?"
    print("=== Testing cot_generator ===")
    res = cg.generate(q, prefer="anthropic")

    safe = {k: v for k, v in res.items() if k not in {"raw", "meta"}}
    print(json.dumps(safe, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
