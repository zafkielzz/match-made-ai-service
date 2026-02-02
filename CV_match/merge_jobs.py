import json
from pathlib import Path

# ===== CONFIG =====
INPUT_DIR = Path("F:\Work_Space\Project\Back_End_Python\CV_match\job_example")   # thư mục chứa nhiều job.json
OUTPUT_FILE = Path("F:\Work_Space\Project\Back_End_Python\CV_match\jobs.json")
# ==================


def load_job_file(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                print(f"[SKIP] {path.name}: not a JSON object")
                return None
            return data
    except Exception as e:
        print(f"[ERROR] {path.name}: {e}")
        return None


def main():
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input directory not found: {INPUT_DIR}")

    jobs = []
    seen_ids = set()

    for file in sorted(INPUT_DIR.glob("*.json")):
        job = load_job_file(file)
        if not job:
            continue

        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            print(f"[SKIP] {file.name}: missing or invalid 'id'")
            continue

        if job_id in seen_ids:
            print(f"[SKIP] {file.name}: duplicate id '{job_id}'")
            continue

        seen_ids.add(job_id)
        jobs.append(job)

    if not jobs:
        raise RuntimeError("No valid job files found.")

    output = {"jobs": jobs}

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[OK] Merged {len(jobs)} jobs → {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
