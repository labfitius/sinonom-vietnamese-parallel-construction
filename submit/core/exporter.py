"""
Exporter Module
Exports aligned parallel corpus to CSV and styled color-coded XLSX.
Packages final submission ZIP structure according to strict project specifications.
"""
import os
import re
import shutil
import zipfile
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from pathlib import Path
from typing import List, Dict, Any
from core.config import COLOR_BLACK, COLOR_RED

_NOM_CHAR_RE = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF\uE000-\uF8FF\U00020000-\U0003FFFF\U000F0000-\U0010FFFF]")
_VI_TOKEN_RE = re.compile(r"([a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+|[^a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ]+)")


class CorpusExporter:
    REQUIRED_COLUMNS = ["img_id", "sentence_id", "SinoNom_Char", "ChuQN_txt"]

    @staticmethod
    def export_csv(records: List[Dict[str, Any]], out_csv_path: Path):
        """
        Export records to CSV using utf-8-sig encoding (BOM) to prevent font issues in Excel.
        Safely handles empty record lists by writing the 4 required header columns.
        """
        out_csv_path = Path(out_csv_path)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            df = pd.DataFrame(columns=CorpusExporter.REQUIRED_COLUMNS)
        else:
            df = pd.DataFrame(records)
            for col in CorpusExporter.REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df = df[CorpusExporter.REQUIRED_COLUMNS]
        df.to_csv(str(out_csv_path), index=False, encoding="utf-8-sig")
        print(f"Exported CSV to {out_csv_path} ({len(df)} rows).")

    @staticmethod
    def export_xlsx(records: List[Dict[str, Any]], row_evals: List[Dict[str, Any]], out_xlsx_path: Path):
        """
        Export records to styled XLSX with black/red color-coding based on alignment evaluation.
        """
        out_xlsx_path = Path(out_xlsx_path)
        out_xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Parallel Corpus"

        # Headers
        headers = CorpusExporter.REQUIRED_COLUMNS
        ws.append(headers)

        header_font = Font(name="Arial", size=11, bold=True, color="FF000000")
        header_fill = PatternFill(start_color="FFE0E0E0", end_color="FFE0E0E0", fill_type="solid")
        thin_border = Border(
            left=Side(style='thin', color='FFD0D0D0'),
            right=Side(style='thin', color='FFD0D0D0'),
            top=Side(style='thin', color='FFD0D0D0'),
            bottom=Side(style='thin', color='FFD0D0D0')
        )

        for col_idx in range(1, 5):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border

        font_black = Font(name="Nom Na Tong", size=11, color=COLOR_BLACK)
        font_meta = Font(name="Arial", size=11, color=COLOR_BLACK)
        inline_nom_black = InlineFont(rFont="Nom Na Tong", sz=12, color=COLOR_BLACK)
        inline_nom_red = InlineFont(rFont="Nom Na Tong", sz=12, color=COLOR_RED, b=True)
        inline_vi_black = InlineFont(rFont="Arial", sz=11, color=COLOR_BLACK)
        inline_vi_red = InlineFont(rFont="Arial", sz=11, color=COLOR_RED, b=True)

        def _build_nom_richtext(nom_text: str, char_statuses: List[tuple]):
            if not nom_text or not char_statuses:
                return nom_text
            if all(ok for _, ok, _ in char_statuses):
                return nom_text
            blocks = []
            st_idx = 0
            cur_ok = None
            buf = []
            for ch in nom_text:
                if _NOM_CHAR_RE.match(ch) and st_idx < len(char_statuses):
                    ok = bool(char_statuses[st_idx][1])
                    st_idx += 1
                else:
                    ok = True
                if cur_ok is None:
                    cur_ok = ok
                    buf.append(ch)
                elif ok == cur_ok:
                    buf.append(ch)
                else:
                    blocks.append(TextBlock(inline_nom_black if cur_ok else inline_nom_red, "".join(buf)))
                    cur_ok = ok
                    buf = [ch]
            if buf:
                blocks.append(TextBlock(inline_nom_black if cur_ok else inline_nom_red, "".join(buf)))
            return CellRichText(*blocks) if blocks else nom_text

        def _build_vi_richtext(qn_text: str, word_statuses: List[bool]):
            if not qn_text or not word_statuses:
                return qn_text
            if all(word_statuses):
                return qn_text
            tokens = _VI_TOKEN_RE.findall(qn_text)
            blocks = []
            w_idx = 0
            cur_ok = None
            buf = []
            for tok in tokens:
                if tok and tok[0].isalpha() and w_idx < len(word_statuses):
                    ok = bool(word_statuses[w_idx])
                    w_idx += 1
                else:
                    ok = True
                if cur_ok is None:
                    cur_ok = ok
                    buf.append(tok)
                elif ok == cur_ok:
                    buf.append(tok)
                else:
                    blocks.append(TextBlock(inline_vi_black if cur_ok else inline_vi_red, "".join(buf)))
                    cur_ok = ok
                    buf = [tok]
            if buf:
                blocks.append(TextBlock(inline_vi_black if cur_ok else inline_vi_red, "".join(buf)))
            return CellRichText(*blocks) if blocks else qn_text

        for row_idx, (rec, eval_info) in enumerate(zip(records, row_evals), start=2):
            img_id = rec.get("img_id", "")
            sent_id = rec.get("sentence_id", "")
            nom_text = str(rec.get("SinoNom_Char", "") or "")
            qn_text = str(rec.get("ChuQN_txt", "") or "")
            char_statuses = eval_info.get("char_statuses", [])
            word_statuses = eval_info.get("word_statuses", [])

            c1 = ws.cell(row=row_idx, column=1, value=img_id)
            c2 = ws.cell(row=row_idx, column=2, value=sent_id)
            c3 = ws.cell(row=row_idx, column=3, value=_build_nom_richtext(nom_text, char_statuses))
            c4 = ws.cell(row=row_idx, column=4, value=_build_vi_richtext(qn_text, word_statuses))

            c1.font = font_meta; c2.font = font_meta
            if not isinstance(c3.value, CellRichText):
                c3.font = font_black
            if not isinstance(c4.value, CellRichText):
                c4.font = font_meta

            for col_idx, cell in enumerate((c1, c2, c3, c4), start=1):
                cell.border = thin_border
                if col_idx in (1, 2):
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

        # Set column widths
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 35
        ws.column_dimensions['C'].width = 40
        ws.column_dimensions['D'].width = 50

        wb.save(str(out_xlsx_path))
        print(f"Exported styled XLSX to {out_xlsx_path} ({len(records)} rows).")

    @staticmethod
    def package_submission_zip(book_dir: Path, zip_dest_path: Path):
        """
        Package book directory into ZIP file.
        """
        book_dir = Path(book_dir)
        zip_dest_path = Path(zip_dest_path)
        zip_dest_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(zip_dest_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(str(book_dir)):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, str(book_dir.parent))
                    zipf.write(full_path, rel_path)
        print(f"Packaged submission archive to {zip_dest_path} ({os.path.getsize(zip_dest_path)} bytes).")
