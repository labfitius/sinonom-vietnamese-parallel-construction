import pandas as pd
from pathlib import Path

df = pd.read_csv("datasets-midterm/audit_dataset_summary.csv", encoding="utf-8-sig")

handwritten = []
non_handwritten = []

for idx, r in df.iterrows():
    stt = int(r["STT_File"])
    name = str(r["Ten_File"])
    code = str(r["Ma_Loai_Sach"])
    notes = str(r["Quy_Cach_Xu_Ly_Theo_Quy_Dinh"])
    nhom = str(r["Nhom_Phan_Loai"])
    
    is_hw = "w" in code.lower()
    item = {
        "stt": stt,
        "name": name,
        "code": code,
        "notes": notes,
        "nhom": nhom
    }
    if is_hw:
        handwritten.append(item)
    else:
        non_handwritten.append(item)

print(f"=== NHOM 1: CHU VIET TAY (HANDWRITTEN) - Tong cong: {len(handwritten)} cuon ===")
for x in handwritten:
    print(f"[{x['stt']:02d}] {x['name']} | Ma: {x['code']}")
    print(f"     Nhom: {x['nhom']}")
    print(f"     Dac diem: {x['notes']}\n")

print(f"\n=== NHOM 2: KHONG PHAI CHU VIET TAY (BAN IN / KHAC MOC / DANH MAY / TEXT) - Tong cong: {len(non_handwritten)} cuon ===")
for x in non_handwritten:
    print(f"[{x['stt']:02d}] {x['name']} | Ma: {x['code']}")
    print(f"     Nhom: {x['nhom']}")
    print(f"     Dac diem: {x['notes']}\n")
