import sys
import os

# Fix encoding issues on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath('.'))

from src.task5_semantic_search import semantic_search

in_domain = [
    'Gợi ý lịch trình du lịch Hạ Long 2 ngày 1 đêm',
    'Những món đặc sản nào nhất định phải thử khi đến Quảng Ninh?',
    'Từ Hà Nội di chuyển đến Hạ Long bằng phương tiện gì?'
]

out_of_domain = [
    'Làm sao để luộc trứng lòng đào ngon?',
    'Giá vé máy bay từ Hồ Chí Minh đi New York',
    'Cách sửa lỗi màn hình xanh trên Windows 11',
    'asdasdqweqweqwe12312312'
]

print('\n--- IN DOMAIN QUERIES ---')
in_scores = []
for q in in_domain:
    res = semantic_search(q, top_k=1)
    score = res[0]['score'] if res else 0
    in_scores.append(score)
    print(f'[{score:.3f}] {q}')

print('\n--- OUT OF DOMAIN QUERIES ---')
out_scores = []
for q in out_of_domain:
    res = semantic_search(q, top_k=1)
    score = res[0]['score'] if res else 0
    out_scores.append(score)
    print(f'[{score:.3f}] {q}')

print('\n--- KẾT LUẬN ---')
min_in = min(in_scores)
max_out = max(out_scores)
print(f'Min In-Domain Score: {min_in:.3f}')
print(f'Max Out-Domain Score: {max_out:.3f}')
if min_in > max_out:
    suggested = (min_in + max_out) / 2
    print(f'Suggested SCORE_THRESHOLD: {suggested:.3f}')
else:
    print('CẢNH BÁO: Điểm In-Domain và Out-Domain bị trùng lấp. Cần xem lại mô hình nhúng!')
