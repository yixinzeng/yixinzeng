import os
# Limit per-process native threads to avoid CPU oversubscription during DB multiprocessing.
# You can override via MATTERGPT_NUMERIC_THREADS=<n>.
_numeric_threads = os.environ.get("MATTERGPT_NUMERIC_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = _numeric_threads
os.environ["OPENBLAS_NUM_THREADS"] = _numeric_threads
os.environ["MKL_NUM_THREADS"] = _numeric_threads
os.environ["NUMEXPR_NUM_THREADS"] = _numeric_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = _numeric_threads
os.environ["BLIS_NUM_THREADS"] = _numeric_threads

import csv
import json
import argparse
import sqlite3
import glob
import hashlib
import time
import multiprocessing as mp
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from slices.utils import splitRun_csv, show_progress, collect_csv
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _compute_file_sha256(file_path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_source_fingerprint(structure_json_path):
    return {
        "source_json_sha256": _compute_file_sha256(structure_json_path),
    }


def _check_structure_database_freshness(db_path, structure_json_path):
    """
    Return (needs_rebuild, reason, source_fingerprint).
    Freshness is defined by JSON content hash, not file path.
    """
    source_fingerprint = _build_source_fingerprint(structure_json_path)
    if not os.path.exists(db_path):
        return True, f"Database '{db_path}' not found.", source_fingerprint

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='structures'")
        if not cursor.fetchone():
            conn.close()
            return True, "Missing table 'structures'.", source_fingerprint

        cursor.execute("SELECT COUNT(*) FROM structures")
        row_count = int(cursor.fetchone()[0])
        if row_count == 0:
            conn.close()
            return True, "Table 'structures' is empty.", source_fingerprint

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='db_meta'")
        if not cursor.fetchone():
            conn.close()
            return True, "Missing table 'db_meta' (old DB format).", source_fingerprint

        cursor.execute("SELECT key, value FROM db_meta")
        meta = {k: v for k, v in cursor.fetchall()}
        conn.close()
    except Exception as e:
        return True, f"Failed to inspect DB: {e}", source_fingerprint

    required_keys = ["source_json_sha256"]
    missing_keys = [k for k in required_keys if k not in meta]
    if missing_keys:
        return True, f"Missing metadata keys: {missing_keys}", source_fingerprint

    if str(meta.get("source_json_sha256", "")) != str(source_fingerprint.get("source_json_sha256", "")):
        return True, "JSON content hash changed.", source_fingerprint

    return False, "Database is up-to-date with novelty JSON.", source_fingerprint


def _prepare_db_record(cif_entry):
    cif_string = cif_entry.get("cif")
    if not cif_string:
        return None, "empty_cif"
    try:
        stru = Structure.from_str(cif_string, "cif")
        finder = SpacegroupAnalyzer(stru)
        primitive_stru = finder.get_primitive_standard_structure()
        spacegroup = finder.get_space_group_number()
        composition = primitive_stru.composition.reduced_formula
        primitive_cif = primitive_stru.to(fmt="cif")
        return (composition, spacegroup, primitive_cif), None
    except Exception as e:
        return None, str(e)


def build_structure_database(structure_json_path, db_path, source_fingerprint=None, db_build_workers=1):
    """
    Build a SQLite structure database from a JSON file containing CIFs.

    The database schema matches the demo workflow and supports novelty checks
    by reduced formula composition and primitive CIF.
    """
    print("开始构建结构数据库（structure_database.db）...")
    print(
        "提示：对于 Materials Project 全集（约 150,000 条结构），"
        "构建过程通常需要 40-50 分钟（取决于 CPU 性能和 I/O 速度），初次构建后新颖性检查无需二次重复构建。"
        "过程中会显示实时进度条，请耐心等待。"
    )

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"已删除旧数据库 '{db_path}'，准备全量重建。")
        except OSError as e:
            print(f"无法删除旧数据库 '{db_path}': {e}")
            exit(1)

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS structures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                composition TEXT,
                spacegroup INTEGER,
                primitive_cif TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS db_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        print("已创建新表 'structures' 和 'db_meta'。")
    except sqlite3.Error as e:
        print(f"数据库连接或创建失败: {e}")
        exit(1)

    try:
        with open(structure_json_path, 'r', encoding='utf-8') as f:
            cifs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Structure JSON file '{structure_json_path}' does not exist.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{structure_json_path}': {e}")
        exit(1)

    total = len(cifs)
    print(f"JSON 文件加载完成，共 {total:,} 条结构记录。")
    if total == 0:
        print("警告：JSON 文件中没有结构数据，跳过数据库构建。")
        conn.close()
        return

    processed = 0
    failed = 0
    insert_sql = "INSERT INTO structures (composition, spacegroup, primitive_cif) VALUES (?, ?, ?)"
    pending_rows = []
    batch_commit_size = 1000
    db_build_workers = max(int(db_build_workers), 1)

    if tqdm is None:
        print("Error: tqdm not installed. Please install it to show progress, e.g., `pip install tqdm`.")
        exit(1)

    parallel_failed = False
    if db_build_workers > 1:
        print(f"使用并行构建：{db_build_workers} 个进程")
        try:
            with tqdm(
                total=total,
                desc="构建数据库",
                unit="结构",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                with mp.Pool(processes=db_build_workers) as pool:
                    for idx, (row, err_msg) in enumerate(
                        pool.imap_unordered(_prepare_db_record, cifs, chunksize=32), start=1
                    ):
                        if row is not None:
                            pending_rows.append(row)
                            processed += 1
                            if len(pending_rows) >= batch_commit_size:
                                cursor.executemany(insert_sql, pending_rows)
                                conn.commit()
                                pending_rows.clear()
                        elif err_msg != "empty_cif":
                            failed += 1
                            if failed <= 10 and err_msg:
                                print(f"\n警告：第 {idx} 条结构处理失败（已跳过）：{err_msg}")
                        pbar.update(1)
        except Exception as e:
            parallel_failed = True
            print(f"并行构建不可用（{type(e).__name__}: {e}），自动回退单进程。")
            cursor.execute("DELETE FROM structures")
            conn.commit()
            processed = 0
            failed = 0
            pending_rows = []

    if db_build_workers <= 1 or parallel_failed:
        with tqdm(
            total=total,
            desc="构建数据库",
            unit="结构",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for idx, cif_entry in enumerate(cifs, start=1):
                row, err_msg = _prepare_db_record(cif_entry)
                if row is not None:
                    pending_rows.append(row)
                    processed += 1
                    if len(pending_rows) >= batch_commit_size:
                        cursor.executemany(insert_sql, pending_rows)
                        conn.commit()
                        pending_rows.clear()
                elif err_msg != "empty_cif":
                    failed += 1
                    if failed <= 10 and err_msg:
                        print(f"\n警告：第 {idx} 条结构处理失败（已跳过）：{err_msg}")
                pbar.update(1)

    if pending_rows:
        cursor.executemany(insert_sql, pending_rows)
        conn.commit()

    if source_fingerprint is None:
        source_fingerprint = _build_source_fingerprint(structure_json_path)
    source_fingerprint = dict(source_fingerprint)
    source_fingerprint["db_built_at_epoch"] = str(int(time.time()))
    cursor.execute("DELETE FROM db_meta")
    for key, value in source_fingerprint.items():
        cursor.execute("INSERT OR REPLACE INTO db_meta (key, value) VALUES (?, ?)", (str(key), str(value)))

    conn.commit()
    print("\n插入完成，正在创建索引以加速后续新颖性查询...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_composition ON structures (composition)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_spacegroup ON structures (spacegroup)")
    conn.commit()
    conn.close()

    print("\n结构数据库构建完成！")
    print(f"   → 成功处理：{processed:,} 条")
    if failed:
        print(f"   → 处理失败：{failed:,} 条（已自动跳过）")
    print(f"   → 数据库保存至：{db_path}")
    print("   → 已创建 composition 索引，后续新颖性检查将显著加速（>100x）")


def ensure_structure_database(structure_json_path, db_path, threads):
    needs_rebuild, reason, fingerprint = _check_structure_database_freshness(db_path, structure_json_path)
    if needs_rebuild:
        print(f"Need rebuild database: {reason}")
        db_build_workers = min(max(1, int(threads)), max(1, os.cpu_count() or 1))
        build_structure_database(
            structure_json_path,
            db_path,
            source_fingerprint=fingerprint,
            db_build_workers=db_build_workers,
        )
    else:
        print(f"Reusing existing database '{db_path}'. {reason}")


def _read_csv_header(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return next(reader)
    except FileNotFoundError:
        print(f"Error: The input CSV file '{csv_path}' does not exist.")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        exit(1)


def process_data(input_csv, output_csv, threads):
    print("Cleaning up old job directories...")
    os.system("rm -rf job_*")

    print("Splitting the input CSV into job files...")
    splitRun_csv(filename=input_csv, threads=threads, skip_header=True)
    show_progress()

    header_in = _read_csv_header(input_csv)
    dynamic_header = header_in + ["novelty"]
    result_header_line = ",".join(dynamic_header) + "\n"

    print(f"Collecting results into '{output_csv}'...")
    collect_csv(
        output=output_csv,
        glob_target="./job_*/result.csv",
        cleanup=False,
        header=result_header_line,
    )

    suspect_files = glob.glob("./job_*/suspect_rows.csv")
    if suspect_files:
        print("Collecting suspect rows into 'suspect_rows.csv'...")
        collect_csv(
            output="suspect_rows.csv",
            glob_target="./job_*/suspect_rows.csv",
            cleanup=True,
            header=result_header_line,
        )
        _dedup_suspect_rows("suspect_rows.csv")
    else:
        os.system("rm -rf job_*")
        print("No suspect rows found.")

    print(f"Results collected into '{output_csv}'.")


def _dedup_suspect_rows(path):
    if not os.path.exists(path):
        return
    print("Removing duplicates from suspect_rows.csv...")
    unique_rows = set()
    temp_dedup_file = "suspect_rows_dedup.csv"

    with open(path, 'r', encoding='utf-8') as fin, \
         open(temp_dedup_file, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for row in reader:
            row_tuple = tuple(row)
            if row_tuple not in unique_rows:
                unique_rows.add(row_tuple)
                writer.writerow(row)

    os.replace(temp_dedup_file, path)
    print("Duplicates removed from suspect_rows.csv.")


def parse_args():
    parser = argparse.ArgumentParser(description="Process structure database and CSV files.")
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to the input CSV file (e.g., ../2_inverse/results.csv).'
    )
    parser.add_argument(
        '--structure_json_for_novelty_check',
        type=str,
        required=True,
        help='Path to the JSON file containing CIF data for novelty checks (e.g., cifs_filtered.json).'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of threads to use for processing (default: 8).'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='results.csv',
        help='Path to the output CSV file (default: results.csv).'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"Error: The input CSV file '{args.input_csv}' does not exist.")
        exit(1)

    if not os.path.isfile(args.structure_json_for_novelty_check):
        print(f"Error: The structure JSON file '{args.structure_json_for_novelty_check}' does not exist.")
        exit(1)

    db_path = "structure_database.db"
    ensure_structure_database(args.structure_json_for_novelty_check, db_path, args.threads)

    process_data(args.input_csv, args.output_csv, args.threads)


if __name__ == "__main__":
    main()
