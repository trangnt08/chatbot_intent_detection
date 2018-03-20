# -*- encoding: utf8 -*-
import re

s = u'Ở Biên Hòa còn hàng không ad.'
print type(s)
rm_junk_mark = re.compile(ur'[?,\.\n]')
normalize_special_mark = re.compile(ur'(?P<special_mark>[\.,\(\)\[\]\{\};!?:“”\"\'/])')
s = normalize_special_mark.sub(u' \g<special_mark> ', s)
s = rm_junk_mark.sub(u'', s)
print s
print type(s)


