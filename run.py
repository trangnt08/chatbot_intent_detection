# -*- coding: utf-8 -*-
__author__ = 'nobita'

# import environment as env
# import accentizer
from flask import Flask, request
import HTMLParser
import uni_big
import ngrams


# accent = accentizer.accentizer()
# accent.fit(env.DATASET, env.DATATEST)

d = {"0": "Giá (sản phẩm, phụ kiện, thay linh kiện,...)", "1": "Hướng dẫn sử dụng (cài phần mềm,...)",
         "2": "Tình trạng sp (còn hay hết)", "3": "So sánh sản phẩm",
         "4": "Chế độ giao hàng", "5": "Chế độ bảo hành", "6": "Khuyến mãi", "7": "Thanh toán (tại cửa hàng, trả góp)",
         "8": "Trả góp","9": "Chất lượng", "10": "Thông tin, chức năng", "11": "Báo khi có máy", "12": "Hủy đơn hàng",
         "14": "Đổi máy, trả máy", "15": "Góp ý", "16": "Đánh giá tích cực", "17": "Phụ kiện (loại gì)", "18": "Khác",
         "19": "Liên hệ, tư vấn", "20": "Đặt trước máy", "21": "PMH có trừ trực tiếp vào máy ko", "22":"Thời gian có máy",
         "23":"Thu mua lại máy","24":"Chào","25":"Muốn mua máy","26":"Thời gian nhận hàng","27":"Địa điểm"}



app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')

@app.route('/', methods = ['GET'])
def homepage():
    return app.send_static_file('index.html')



@app.route('/intent', methods=['POST'])
def process_request():
    data = request.form['data']
    data = HTMLParser.HTMLParser().unescape(data)
    kq = uni_big.predict_ex(data)
    # kq = ngrams.predict_ex(data)
    return d[kq]
    # return accent.predict(data)

if __name__ == '__main__':
    app.run('0.0.0.0', port=9900)
