def cal(x, y):
    global count
    # print(count)
    count += 1
    if (x > 1):
        cal(x-2, y+2)
        print(y)
    # print('Called')


count = 1
cal(4, 5)
