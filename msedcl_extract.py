"""
MSEDCL Tariff PDF Extractor
============================
Project structure assumed:

    elec_analysis/
    ├── data/
    │   ├── pdfs/               ← put all your MERC PDF files here
    │   └── csv/                ← CSVs will be saved here (auto-created)
    ├── msedcl_extract.py       ← THIS script (run from elec_analysis/)
    └── msedcl_analyse.py       ← analysis script (next step)

RUN FROM inside elec_analysis/ folder:
    python msedcl_extract.py

DEPENDENCIES:
    pip install pdfplumber pandas
    +
    +
    """

import pdfplumber
import pandas as pd
import os
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
#  PROJECT PATHS  (relative to elec_analysis/ folder)
# ══════════════════════════════════════════════════════════════════
BASE_DIR    = Path(__file__).parent          # elec_analysis/
PDF_DIR     = BASE_DIR / "data"              # elec_analysis/data/
OUTPUT_DIR  = BASE_DIR / "data" / "csv"     # elec_analysis/data/csv/

# ══════════════════════════════════════════════════════════════════
#  ▶  EDIT THIS — map each PDF filename to its FY label
#     Keys   = exact filename inside elec_analysis/data/
#     Values = FY label used in the output CSVs
# ══════════════════════════════════════════════════════════════════
PDF_FILES = {
    "FY 2024-25.pdf"  : "2024-25",
    "FY 2023-24.pdf"  : "2023-24",
    "FY 2022-23.pdf"  : "2022-23",
    "FY 2019-20.pdf"  : "2019-20",
    "FY 2018-19.pdf"  : "2018-19",
    "FY 2016-17.pdf"  : "2016-17",
    "FY 2014-15.pdf"  : "2014-15",
    "FY 2012-13.pdf"  : "2012-13",
    "FY 2009-10.pdf"  : "2009-10",
}


# ══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def clean_value(val):
    """Strip whitespace/newlines; return None for blank/dash cells."""
    if val is None:
        return None
    val = str(val).replace('\n', ' ').strip()
    return None if val in ('', '-', '--', 'N/A', 'NA', 'nil', 'Nil', 'NIL') else val


def is_numeric(val):
    if val is None:
        return False
    try:
        float(str(val).replace(',', ''))
        return True
    except ValueError:
        return False


def parse_numeric(val):
    try:
        return float(str(val).replace(',', ''))
    except (ValueError, TypeError):
        return None


def detect_section(text):
    """Classify a page into a tariff section using keyword matching."""
    t = text.lower()
    checks = [
        ('LT_Residential',    ['lt i(a)', 'lt i(b)', 'lt-i', 'residential', 'bpl']),
        ('LT_Commercial',     ['lt ii', 'non-residential', 'commercial']),
        ('LT_Agriculture',    ['lt iv', 'agriculture', 'agricultural', 'ag un-metered']),
        ('LT_Industry',       ['lt v', 'lt-v', 'lt industry', 'powerloom']),
        ('LT_StreetLight',    ['lt vi', 'street light', 'street lighting']),
        ('LT_PublicServices', ['lt vii', 'public service']),
        ('LT_EV',             ['lt viii', 'electric vehicle charging']),
        ('HT_Industry',       ['ht i(a)', 'ht i(b)', 'ht industry']),
        ('HT_Commercial',     ['ht ii', 'ht commercial']),
        ('HT_Agriculture',    ['ht v', 'ht agriculture']),
        ('HT_Railways',       ['ht iii', 'railway', 'metro', 'monorail']),
        ('HT_PWW',            ['ht iv', 'public water works', 'pww']),
        ('HT_GroupHousing',   ['ht vi', 'group housing', 'housing societ']),
        ('HT_PublicServices', ['ht viii', 'ht public service']),
        ('HT_EV',             ['ht ix', 'ht electric vehicle']),
    ]
    for label, keywords in checks:
        if any(k in t for k in keywords):
            return label
    return 'General'


def find_tariff_pages(pdf):
    """Return list of (page_index, page_text) for pages that have tariff tables."""
    KEYWORDS = [
        'fixed charge', 'energy charge', 'demand charge', 'wheeling charge',
        'rs/kwh', 'rs/kvah', 'rs/kva/mth', 'rs/hp/mth', 'rs/conn/mth', 'rs/kw/mth',
        'summary of lt', 'summary of ht', 'lt tariff', 'ht tariff',
        'approved tariff', 'retail tariff', 'tariff schedule',
        'table 7-', 'table 6-', 'table 5-',
    ]
    result = []
    for i, page in enumerate(pdf.pages):
        try:
            text = page.extract_text() or ''
            if any(k in text.lower() for k in KEYWORDS) and page.extract_tables():
                result.append((i, text))
        except Exception:
            continue
    return result


def extract_table_rows(table, fy, section, page_num):
    """Parse one pdfplumber table into a list of structured dicts."""
    rows_out = []

    # ── Detect header row ──
    header_idx = None
    for idx, row in enumerate(table[:7]):
        joined = ' '.join(str(c).lower() for c in row if c)
        if any(h in joined for h in ['fixed', 'energy', 'demand', 'wheeling']):
            header_idx = idx
            break

    # ── Map column positions from header ──
    cat_col = unit_col = fixed_col = energy_col = wheeling_col = None
    if header_idx is not None:
        for ci, cell in enumerate(table[header_idx]):
            if not cell:
                continue
            cl = str(cell).lower()
            if 'categor' in cl or 'particular' in cl:
                cat_col = ci
            elif 'unit' in cl:
                unit_col = ci
            elif 'fixed' in cl or 'demand' in cl:
                fixed_col = ci
            elif 'energy' in cl:
                energy_col = ci
            elif 'wheeling' in cl:
                wheeling_col = ci

    cat_col  = cat_col  if cat_col  is not None else 0
    unit_col = unit_col if unit_col is not None else 1

    # ── Parse data rows ──
    current_category = ''
    for row in table[(header_idx + 1 if header_idx is not None else 0):]:
        cleaned = [clean_value(c) for c in row]
        if all(c is None for c in cleaned):
            continue

        cat_val  = cleaned[cat_col]  if cat_col  < len(cleaned) else None
        unit_val = cleaned[unit_col] if unit_col < len(cleaned) else None

        if cat_val and len(cat_val) > 2:
            current_category = cat_val

        # ── Extract numeric charge values ──
        fixed_val = energy_val = wheeling_val = None

        if fixed_col is not None and fixed_col < len(cleaned):
            fixed_val = parse_numeric(cleaned[fixed_col])
        if energy_col is not None and energy_col < len(cleaned):
            energy_val = parse_numeric(cleaned[energy_col])
        if wheeling_col is not None and wheeling_col < len(cleaned):
            wheeling_val = parse_numeric(cleaned[wheeling_col])

        # Fallback: pull numbers by position when column detection missed
        if fixed_val is None and energy_val is None:
            nums = [
                (ci, parse_numeric(c))
                for ci, c in enumerate(cleaned)
                if ci not in (cat_col, unit_col) and is_numeric(c)
            ]
            if len(nums) >= 3:
                fixed_val, energy_val, wheeling_val = nums[0][1], nums[1][1], nums[2][1]
            elif len(nums) == 2:
                fixed_val, energy_val = nums[0][1], nums[1][1]
            elif len(nums) == 1:
                energy_val = nums[0][1]

        if fixed_val is None and energy_val is None and wheeling_val is None:
            continue

        rows_out.append({
            'FY'             : fy,
            'Section'        : section,
            'Page'           : page_num + 1,
            'Category'       : current_category or cat_val or 'Unknown',
            'Units'          : unit_val,
            'Fixed_Charge'   : fixed_val,
            'Energy_Charge'  : energy_val,
            'Wheeling_Charge': wheeling_val,
            'Raw_Row'        : ' | '.join(str(c) for c in cleaned if c is not None),
        })

    return rows_out


# ══════════════════════════════════════════════════════════════════
#  PER-PDF EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_pdf(pdf_path: Path, fy: str) -> pd.DataFrame:
    print(f"\n{'─'*60}")
    print(f"  📄 {pdf_path.name}  →  FY {fy}")
    print(f"{'─'*60}")

    if not pdf_path.exists():
        print(f"  ⚠️  File not found — skipping.")
        return pd.DataFrame()

    all_rows = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"  Pages : {len(pdf.pages)}")
            tariff_pages = find_tariff_pages(pdf)
            print(f"  Tariff pages detected: {len(tariff_pages)}")

            for page_idx, page_text in tariff_pages:
                section = detect_section(page_text)
                try:
                    tables = pdf.pages[page_idx].extract_tables()
                except Exception as e:
                    print(f"    ⚠️  Page {page_idx+1} error: {e}")
                    continue

                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    rows = extract_table_rows(table, fy, section, page_idx)
                    if rows:
                        all_rows.extend(rows)
                        print(f"    ✔  Page {page_idx+1} [{section}] → {len(rows)} rows")

    except Exception as e:
        print(f"  ❌ Could not open PDF: {e}")
        return pd.DataFrame()

    if not all_rows:
        print("  ⚠️  No tariff data found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.drop_duplicates(subset=['FY', 'Category', 'Fixed_Charge', 'Energy_Charge'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  ✅ {len(df)} rows extracted")
    return df


# ══════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  MSEDCL Tariff Extractor")
    print(f"  PDF source : {PDF_DIR}")
    print(f"  CSV output : {OUTPUT_DIR}")
    print(f"{'═'*60}")

    # Show what PDFs are actually present
    found = sorted(PDF_DIR.glob("FY *.pdf")) if PDF_DIR.exists() else []
    print(f"\n  PDFs found in data/pdfs/  ({len(found)} files):")
    for f in found:
        print(f"    📄 {f.name}")

    all_dfs = []
    summary = []

    for filename, fy in PDF_FILES.items():
        pdf_path = PDF_DIR / filename
        df = extract_pdf(pdf_path, fy)

        if not df.empty:
            safe_fy  = fy.replace('-', '_')
            csv_path = OUTPUT_DIR / f"msedcl_tariff_{safe_fy}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  💾 Saved → data/csv/msedcl_tariff_{safe_fy}.csv")
            all_dfs.append(df)
            summary.append({'FY': fy, 'File': filename, 'Rows': len(df), 'Status': '✅ Done'})
        else:
            summary.append({'FY': fy, 'File': filename, 'Rows': 0, 'Status': '⚠️  No data / Missing'})

    # Master combined CSV (feeds directly into analysis script)
    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
        master_path = OUTPUT_DIR / "msedcl_ALL_YEARS_master.csv"
        master.to_csv(master_path, index=False)
        print(f"\n  📦 Master CSV saved → data/csv/msedcl_ALL_YEARS_master.csv")
        print(f"     Total rows across all years: {len(master)}")

    # Summary
    print(f"\n{'═'*60}")
    print(f"  EXTRACTION SUMMARY")
    print(f"{'═'*60}")
    for s in summary:
        print(f"  {s['Status']}  FY {s['FY']}  |  {s['Rows']:>4} rows  |  {s['File']}")

    print(f"\n  Final structure:")
    print(f"  elec_analysis/")
    print(f"  └── data/")
    print(f"      ├── (PDFs)    ← your input PDFs live here directly")
    print(f"      └── csv/      ← generated output")
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            print(f"          📄 {f.name}  ({f.stat().st_size/1024:.1f} KB)")

    print(f"\n  ✅ Extraction complete! Next step: run msedcl_analyse.py")


if __name__ == '__main__':
    run()
