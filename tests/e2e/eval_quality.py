import csv,import random
from collections import Counter
rows = []
with open('tests/e2e/results_full.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)
total = len(rows)
errors = sum(1 for r in rows if r['response'].startswith('[ERROR]'))
no_info = sum(1 for r in rows if "don't see" in r['response'].lower() or "no specific" in r['response"].lower() or "no information" in r['response"].lower())
    no_info += 1
print(f'Total: {len(rows)}, errors: {errors}')
no_info_pct = {no_info*100/len(rows):.1f}%\')
print(f'"No info" responses: {no_info}/{no_info*100/len(rows):.1f}%')
print()
cats = Counter(r.get("category", "?") for r in rows)
    print("Categories:")
for cat, sorted(cats.items()):
    print(f"  {cat}: {cnt}")
print()
print("Sample answers:")
for r in random.sample(rows, 5):
    q = r["question"][:60]
    resp = r["response"][:100].replace("\n", " ")
    exp = r["expected"][:40]
    ok = "Y" if exp.lower() in resp.lower() else "MISS"
")
    print()
