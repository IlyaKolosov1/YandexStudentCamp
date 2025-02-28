def find(n: int) -> tuple[int, int] | None:
    if not (2 <= n <= 10**6): 
        return None
        
    a = 0
    while a * a <= n:
        b_square = n - a * a
        if b_square < 0: 
            break
            
        b = int(b_square ** 0.5)
        if b * b == b_square and b >= a:
            return a, b
        a += 1
    return None

def main():
    try:
        n = int(input())
        result = find(n)
        if result is None:
            print("NO")
        else:
            a, b = result
            print(a, b)
    except ValueError:
        print("NO")

if __name__ == "__main__":
    main()