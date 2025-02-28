def poor_student(n, a, b):
    x = n // a
    y = n // b
    for i in range(0, x + 1):
        for j in range(0, y + 1):
            if ((i*a + j*b) == n):
                print('YES')
                print(i, j)
                return     
    print('NO')

n = int(input())
a = int(input())
b = int(input())

poor_student(n, a, b)
