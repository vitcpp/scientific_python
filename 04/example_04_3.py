import re

s = "this is my email vit@domain.ru"

m = re.search("([\w\.-]+)@([\w\.-]+)", s)
if m:
    print(m.group(0))
    print(m.group(1))
    print(m.group(2))
