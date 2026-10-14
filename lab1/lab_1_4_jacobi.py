import math
import os
import sys
from lab_1_1_LU import matrix_mul


def read_input(filename):
    """Чтение симметричной матрицы A и точности eps из файла"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден!")

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    A = []
    for i in range(1, n + 1):
        row = [float(x) for x in lines[i].split()]

        if len(row) != n:
            raise ValueError(f"Ошибка в строке {i}: ожидалось {n} элементов")
        
        A.append(row)

    return n, A


def is_symmetric(A, tol=1e-6):
    """Проверка симметричности матрицы A = A^T"""
    
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
            
    return True


def max_off_diagonal(A):
    """Поиск максимального по модулю внедиагонального элемента"""

    n = len(A)
    max_val = 0.0
    l, m = 0, 1

    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                l, m = i, j

    return l, m, max_val


def off_diagonal_norm(A):
    """Сумма квадратов внедиагональных элементов"""

    n = len(A)
    s = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            s += A[i][j] ** 2

    return math.sqrt(s)


def jacobi_rotation(A, eps):
    """Метод вращения Якоби"""

    n = len(A)
    A_k = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    iterations = 0
    max_iter = 10000

    while True:
        l, m, max_val = max_off_diagonal(A_k)
        if off_diagonal_norm(A_k) < eps or iterations >= max_iter:
            break

        iterations += 1

        p = (A_k[l][l] - A_k[m][m]) / (2.0 * A_k[l][m])
        sgn_p = 1.0 if p >= 0 else -1.0
        t = sgn_p / (abs(p) + math.sqrt(p**2 + 1.0))
        c = 1.0 / math.sqrt(1.0 + t**2)
        s = t * c

        A_next = [row[:] for row in A_k]

        A_next[l][l] = c * c * A_k[l][l] + 2.0 * c * s * A_k[l][m] + s * s * A_k[m][m]
        A_next[m][m] = s * s * A_k[l][l] - 2.0 * c * s * A_k[l][m] + c * c * A_k[m][m]
        A_next[l][m] = 0.0
        A_next[m][l] = 0.0

        for i in range(n):
            if i != l and i != m:
                A_next[i][l] = c * A_k[i][l] + s * A_k[i][m]
                A_next[l][i] = A_next[i][l]
                A_next[i][m] = -s * A_k[i][l] + c * A_k[i][m]
                A_next[m][i] = A_next[i][m]

        A_k = A_next

        for i in range(n):
            v_il = V[i][l]
            v_im = V[i][m]
            V[i][l] = c * v_il + s * v_im
            V[i][m] = -s * v_il + c * v_im

    eigenvalues = [A_k[i][i] for i in range(n)]
    return eigenvalues, V, iterations


def format_matrix(M, name=""):
    res = []

    if name:
        res.append(f"--- {name} ---")
    for row in M:
        res.append("  [" + ", ".join(f"{x:12.6f}" for x in row) + "]")
    return "\n".join(res)


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "./1_4_tests/1.txt"
    n, A = read_input(filename)
    eps = 0.01

    print("==================================================")
    print("ЛАБОРАТОРНАЯ РАБОТА № 1.4: МЕТОД ВРАЩЕНИЙ ЯКОБИ")
    print("==================================================\n")

    print(format_matrix(A, "Исходная матрица A"))
    print()

    if not is_symmetric(A):
        print("[ОШИБКА] Исходная матрица не является симметричной! Метод Якоби не применим.")
        return

    eigenvalues, V, iterations = jacobi_rotation(A, eps)

    print("==================================================")
    print(f"1. ЗАДАННАЯ ТОЧНОСТЬ ВЫЧИСЛЕНИЙ ε = {eps}")
    print(f"   Количество выполненных итераций: {iterations}")
    print("==================================================\n")

    print("2. НАЙДЕННЫЕ СОБСТВЕННЫЕ ЗНАЧЕНИЯ (λ):")
    for i in range(n):
        print(f"   λ_{i+1} = {eigenvalues[i]:12.6f}")
    print("\n" + "="*50 + "\n")

    print("3. НАЙДЕННЫЕ СОБСТВЕННЫЕ ВЕКТОРЫ И МАТРИЦА V:")
    print(format_matrix(V, "Матрица собственных векторов V (столбцы — векторы v)"))
    print()
    for j in range(n):
        v_j = [V[i][j] for i in range(n)]
        print(f"   Собственный вектор v^({j+1}) для λ_{j+1}: [{', '.join(f'{x:9.6f}' for x in v_j)}]")
    print("\n" + "="*50 + "\n")

    print("4. ПРОВЕРКА A · V = V · Λ:")
    AV = matrix_mul(A, V)

    Lambda = [[eigenvalues[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    VL = matrix_mul(V, Lambda)

    print(format_matrix(AV, "Матрица A · V"))
    print()
    print(format_matrix(VL, "Матрица V · Λ"))

    max_diff = max(abs(AV[i][j] - VL[i][j]) for i in range(n) for j in range(n))
    print(f"\n[РЕЗУЛЬТАТ] Максимальное расхождение ||A·V - V·Λ|| = {max_diff:.6e}")
    print(f"   Совпадение A · V = V · Λ с точностью ε: {max_diff <= eps}")
    print("==================================================")


if __name__ == "__main__":
    main()