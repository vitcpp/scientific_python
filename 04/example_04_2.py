# Проблемный код

f = None

try:
    f = open("data.dat", "rt")
    content = f.read()
    print(content)
#except Exception as e:
#     print("ERROR: Ошибка чтения файла ({})".format(e))
finally:
    print("FINALLY!!!")
    if f != None:
        f.close()
