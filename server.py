# -*- encoding: utf-8 -*-

from flask import Flask, request, flash, render_template
from sklearn.externals import joblib
import uni_big
from io import open


app = Flask('crf')
d = {"0": "Giá (sản phẩm, phụ kiện, thay linh kiện,...)", "1": "Cập nhật phần mềm hệ thống (cài phần mềm,...)",
         "2": "Tình trạng sp (còn hay hết)", "3": "So sánh 2 sản phẩm",
         "4": "Chế độ giao hàng", "5": "Chế độ bảo hành", "6": "Khuyến mãi", "7": "Thanh toán (tại cửa hàng, trả góp)",
         "8": "Trả góp",
         "9": "Chất lượng", "10": "Thông tin, chức năng", "11": "Báo khi có máy", "12": "Hủy đơn hàng", "14": "Đổi máy",
         "15": "Góp ý", "16": "Đánh giá tích cực", "17": "Phụ kiện (loại gì)", "18": "Khác", "19": "Tư vấn",
         "20": "Đặt trước máy",
         "21": "PMH có trừ trực tiếp vào máy ko"}
with open('home.html', 'r', encoding='utf-8') as f:
	data1 = f.read()

@app.route('/',methods = ['GET','POST'])
def homepage():
    print "aa"
    try:
        error = None
        if request.method == "GET":
            print "get"
            return data1
        if request.method == "POST":
            print "post "
            data2 = request.get_data()
            print "b", data2
            print "cc"
            kq = uni_big.predict_ex(data2)
            print 'kq ',kq
            return d[kq]
    except:
        return 'err'
	return data

if __name__ == '__main__':
    app.run(port=8080)