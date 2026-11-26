Hướng dẫn cách đọc file XLSX:

1. ned_summary.xlsx: chứa thống kê tỉ lệ lỗi của các file sách
2. Các file XLSX còn lại chứa nhiều cột, trong đó các cột chính gồm:
     + Patch ID: xác định tên sách, trang sách và dòng tương ứng
     + Vietnamese: dịch âm Quốc ngữ đã dóng hàng
     + Label: label hiện tại (đã kết hợp voting nhiều mô hình và hậu chỉnh sửa edit-distance). Lưu ý: trong lúc hậu chỉnh sửa, có nhiều chữ cái được các mô hình đồng thuận với conf cao nhưng lại không có âm giống với label tiếng Việt. Sau khi kiểm tra, những trường hợp này do âm tiếng Việt bị lỗi chính tả. Do đó, khi gặp những trường hợp như vậy, chương trình chấp nhận kết quả voting (correctness=1) dù lệch âm.
     + Correctness: chuỗi nhị phân có chiều dài bằng với số lượng chữ cái của dòng. 1: chữ Nôm và âm QN có trong từ điển; 0: lỗi insertion hoặc substitution

3. Trong mỗi folder quyển sách (<book-name>/images/), có Label.txt và char_det.txt:
     + Label.txt: label textline cấp độ dòng
     + char_det.txt: label characters
