import requests
import re
# data = {'data':'dt nay bao gia bn'}
# r = requests.post('http://topica.ai:9339/accent', data=data)
# result = r.content
# try:
#     result = unicode(result)
# except:
#     result = unicode(result, encoding='utf-8')
# result = result.split(u'\n')[1]
# print result

def accent(req):
    data = {'data': req}
    r = requests.post('http://topica.ai:9339/accent', data=data)
    result = r.content
    try:
        result = unicode(result)
    except:
        result = unicode(result, encoding='utf-8')
    result = result.split(u'\n')[1]
    return result

def regex_phone_number():
    str = "sdt cua toi la 0975025641 va 01662486166 hoa 0123 456 789 voi 097 222 4444"
    a = [s for s in str.split() if s.isdigit() and len(s)>=10]
    reg = re.findall("\d{2,4}\D{0,3}\d{3}\D{0,3}\d{3,4}",str)
    print reg
    # print a
    for x in reg:
        print x
        str = str.replace(x,"phone_number")
    print str

def regex_link():
    url = 'Hello W http://example.com More Examples a href "http://example2.com" Even More Examples'
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url)
    print urls


if __name__ == '__main__':
    s = accent("dt nay gia bao nhieu")
    print s
    # regex_phone_number()
    # regex_link()