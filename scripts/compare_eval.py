# -*- coding: utf-8 -*-
"""전이학습 전(base) vs 후(FT) CER 비교 표 출력.
사용: python scripts/compare_eval.py <before.json> <after.json>"""
import json, sys

def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return d

b = load(sys.argv[1]); a = load(sys.argv[2])

def pct(x):
    return f"{x*100:.2f}%" if isinstance(x,(int,float)) else str(x)

def rel(bv, av):
    if not bv: return "-"
    return f"{(av-bv)/bv*100:+.1f}%"

print(f"\n{'='*64}")
print(f"  전이학습 전/후 CER 비교  (test split, 동일 셋)")
print(f"{'='*64}")
print(f"  before: {sys.argv[1]}")
print(f"  after : {sys.argv[2]}")
print(f"\n{'구분':<10}{'전(base)':>12}{'후(FT)':>12}{'상대개선':>12}")
print("-"*46)
bo, ao = b.get("overall_cer"), a.get("overall_cer")
print(f"{'전체':<10}{pct(bo):>12}{pct(ao):>12}{rel(bo,ao):>12}")
bc, ac = b.get("by_category",{}) or {}, a.get("by_category",{}) or {}
for cat in sorted(set(bc)|set(ac)):
    print(f"{cat:<10}{pct(bc.get(cat)):>12}{pct(ac.get(cat)):>12}{rel(bc.get(cat),ac.get(cat)):>12}")
# 의료진 통합(의사+간호사) 참고치는 by_category 가중평균이 아니라 표기만 분리
print("-"*46)
print("※ 의료진=의사+간호사, 환자 별도. 사장님 환경(의사↔환자) 핵심은 '의사','환자' 행.")
