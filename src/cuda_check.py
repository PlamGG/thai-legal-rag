import torch
print(torch.cuda.is_available())  
print(torch.cuda.get_device_name(0)) 
print(torch.version.cuda) 

# import pickle
# import sys

# # โหลด passages
# try:
#     with open("./data/passages.pkl", "rb") as f:
#         passages = pickle.load(f)
# except FileNotFoundError:
#     print("Error: The file 'passages.pkl' was not found. Please make sure you have run the data preparation scripts first.")
#     sys.exit(1)

# print(f"Total passages: {len(passages)}\n")

# # ดู 3 ตัวอย่างแรก
# for i, p in enumerate(passages[:3]):
#     print(f"Passage {i+1}:")
#     print("ID:", p.get("id"))
#     print("Text:", p.get("text")[:200], "...")  # แสดงแค่ 200 ตัวอักษรแรก
#     print("Metadata:", p.get("metadata"))
#     print("-" * 50)

# # ส่วนที่เพิ่มเข้ามา: หาและนับจำนวนประเภทของ law_title
# unique_law_titles = set()
# for p in passages:
#     metadata = p.get("metadata", {})
#     law_title = metadata.get("law_title")
#     if law_title:
#         unique_law_titles.add(law_title)

# print("\n" + "=" * 50)
# print("สรุปข้อมูล 'law_title'")
# print(f"จำนวนประเภทของกฎหมายทั้งหมด: {len(unique_law_titles)} ประเภท")
# print("รายการประเภทของกฎหมาย:")
# # จัดเรียงตามตัวอักษรเพื่อให้อ่านง่าย
# for title in sorted(list(unique_law_titles)):
#     print(f"- {title}")
# print("=" * 50)

# # ดู field ทั้งหมดใน metadata ของ passages (โค้ดเดิมของคุณ)
# metadata_keys = set()
# for p in passages:
#     metadata_keys.update(p.get("metadata", {}).keys())
# print("\nAll metadata keys found:", metadata_keys)
