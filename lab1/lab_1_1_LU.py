import os
import sys
from fractions import Fraction


def parse_number(s):
    """Преобразование строки в дробь"""

    s = s.strip()

    if '/' in s:
        num, den = s.split('/')
        return Fraction(int(num), int(den))
    elif '.' in s:
        return Fraction(s)
    else:
        return Fraction(int(s))


def format_matrix(M):
    """Вывод матрицы"""

    res = []

    for row in M:
        res.append(' [' + ', '.join(f'{str(x):>10}' for x in row) + ']')

    return '\n'.join(res)


def format_vector(v):
    """Вывод вектора"""

    res = []
    res.append(' [' + ', '.join(str(x) for x in v) + ']')

    return '\n'.join(res)


def read_input(filename):
    """Чтение из файла"""

    filename = '1_1_tests/' + filename

    if not os.path.exists(filename):
        raise FileNotFoundError('Файл не найден!')

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError('Файл пуст!')

    n = int(lines[0])
    A = []

    for i in range(1, n+1):
        row = [parse_number(x) for x in lines[i].split()]

        if len(row) != n:
            raise ValueError(f'Ошибка в строке {i}')

        A.append(row)

    b = [parse_number(x) for x in lines[n+1].split()]

    if len(b) != n:
        raise ValueError('Ошибка в векторе b')

    return n, A, b


def matrix_mul(A, B):
    """Матричное умножение"""

    n = len(A)
    m = len(A[0])
    p = len(B[0])

    C = [[Fraction(0) for _ in range(p)] for _ in range(n)]

    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]

    return C


def lu_decomposition(A):
    """LU-разложение"""

    n = len(A)
    U = [row[:] for row in A]
    L = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]
    P = [[Fraction(1 if i == j else 0) for j in range(n)] for i in range(n)]

    swaps_count = 0

    for k in range(n):
        max_row = k
        max_val = abs(U[k][k])

        for i in range(k+1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i

        if max_val == 0:
            raise ValueError('Вырожденная матрица')

        if max_row != k:
            U[k], U[max_row] = U[max_row], U[k]
            P[k], P[max_row] = P[max_row], P[k]
            swaps_count += 1

            for j in range(k):
                L[k][j], L[max_row][j] = L[max_row][j], L[k][j]

        for i in range(k+1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor

            for j in range(k, n):
                U[i][j] -= factor * U[k][j]

    return L, U, P, swaps_count


def solve_lu(L, U, P, b):
    """Решение СЛАУ Ax = b через L(Ux) = Pb"""

    n = len(L)
    Pb = [Fraction(0) for _ in range(n)]

    for i in range(n):
        for j in range(n):
            Pb[i] += P[i][j] * b[j]

    z = [Fraction(0) for _ in range(n)]

    for i in range(n):
        s = sum(L[i][j] * z[j] for j in range(i))
        z[i] = Pb[i] - s

    x = [Fraction(0) for _ in range(n)]

    for i in range(n-1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (z[i] - s) / U[i][i]

    return x


def invert_lu(L, U, P):
    """Вычисление обратной матрицы A^(-1)"""

    n = len(L)
    inv = [[Fraction(0) for _ in range(n)] for _ in range(n)]

    for j in range(n):
        e = [Fraction(1 if i == j else 0) for i in range(n)]
        col = solve_lu(L, U, P, e)

        for i in range(n):
            inv[i][j] = col[i]

    return inv


def det_lu(U, swaps_count):
    """Det(A)"""

    det = Fraction(1)

    for i in range(len(U)):
        det *= U[i][i]

    return (-1) ** swaps_count * det


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else 'input_1_1.txt'

    n, A, b = read_input(filename)

    print(format_matrix(A))
    print()
    print(format_vector(b))
    print()

    L, U, P, swaps_count = lu_decomposition(A)

    print('1. L and U')
    print(format_matrix(L))
    print()
    print(format_matrix(U))
    print()
    print(format_matrix(P))
    print()

    lu = matrix_mul(L, U)
    print('2. L * U')
    print(format_matrix(lu))
    print()

    x = solve_lu(L, U, P, b)
    print('3. решение Ax = b')
    print(format_vector(x))
    print()

    A_inv = invert_lu(L, U, P)
    print('4. обратная матрица')
    print(format_matrix(A_inv))
    print()

    det_A = det_lu(U, swaps_count)
    print('5. определитель матрицы A')
    print(f'det(A) = {det_A}')
    print()

    AA_inv = matrix_mul(A, A_inv)
    print('6. A * A^(-1) = E')
    print(format_matrix(AA_inv))
    print()

    PA = matrix_mul(P, A)
    check = lu == PA
    print('7. L * U == P * A')
    print(format_matrix(PA))
    print()
    print(f'результат: {check}')
    print()


if __name__ == '__main__':
    main()
