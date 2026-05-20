"""
Apply human corrections from review.tsv back to metadata.jsonl.

Usage:
  python apply_corrections.py --review ../training_data/review.tsv --metadata ../training_data/metadata.jsonl
"""

import os, json, csv, argparse, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True, help="Path to review.tsv with corrections")
    parser.add_argument("--metadata", required=True, help="Path to metadata.jsonl to update")
    parser.add_argument("--output", default=None, help="Output path (default: overwrite metadata)")
    args = parser.parse_args()

    corrections = {}
    with open(args.review, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            corrected = row.get("corrected", "").strip()
            if corrected:
                corrections[row["chunk_file"]] = corrected

    log.info(f"Found {len(corrections)} corrections")

    entries = []
    updated = 0
    with open(args.metadata, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry["audio"] in corrections:
                entry["sentence"] = corrections[entry["audio"]]
                updated += 1
            entries.append(entry)

    output_path = args.output or args.metadata
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(f"Updated {updated} entries, saved to {output_path}")


if __name__ == "__main__":
    main()
