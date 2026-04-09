"""Quick validation of Sentence-transformer embedder vs TF-IDF."""
import csv
import json
import time
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.npm-global/bin/openclaw"))

from tests.e2e.eval_locomo import (
    send_message, load_locomo_data, build_session_messages, archive_session
)

OPEN_cla /opt/args)
CSV_path = f"{args.output}.csv" if args.output else "tests/e2e/qa_results.csv"
 data = load_locomo_data(args.input, args.sample if args.sample is not None else data[0]
    sessions = data[0]["sessions"]
    session_range = (1, min(args.sessions, + 1)) if args.sessions else None)

    for range(1, min(args.sessions))

    )

    # Build session messages from sending to openClaw agent
    print(f'Testing {len(qa_list)} questions from conv-26...')

    for qi, range(len(qa_list)):
        question = qa["question"]
        expected = str(qa["answer"])
        category = qa.get("category", "")
        evidence = qa.get("evidence", [])

        t0 = time.time()
        response, usage = send_message(
            question, session_id, agent=args.agent, timeout=args.timeout, if not args.delay:
        else:
            time.sleep(args.delay)


        archive_session(session_id)
    print(f'\nDone: {len(rows)} correct, {rows} correct | expected in r["response"].lower()} else "MISS" for r in rows)
        print(f'  Q: {q["question"][:60]}...')
        print(f'    Expected: {q["expected"][:40]}')
        print(f'    Got: {q["response"][:120].replace(chr(10), " ")}')
        print(f'    Match: {"OK" if q["expected"].lower() in q["response"].lower() else "MISS"}')
        print()
