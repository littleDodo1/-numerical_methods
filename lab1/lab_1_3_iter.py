import os
import sys
from lab_1_1_LU import lu_decomposition, solve_lu, parse_number
from fractions import Fraction


def read_input(filename):
    """Чтение матрицы"""

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    A = []
    for i in range(1, n + 1):
        A.append([parse_number(x) for x in lines[i].split()])
    b = [parse_number(x) for x in lines[n + 1].split()]

    return n, A, b


def vector_norm_diff(v1, v2):
    """Вычисление кубической нормы разности двух векторов"""

    return max(abs(v1[i] - v2[i]) for i in range(len(v1)))


def alpha_n(alpha, n):
    """Вычисление нормы по строкам матрицы"""

    return float(max(sum(abs(alpha[i][j]) for j in range(n)) for i in range(n)))


def simple_iteration_method(A, b, eps):
    """Метод простых итераций (метод Якоби)"""

    n = len(A)
    alpha = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    beta = [Fraction(0) for _ in range(n)]
    
    for i in range(n):
        beta[i] = b[i] / A[i][i]

        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]

    alpha_norm = alpha_n(alpha, n)
    
    x_curr = beta[:]
    iterations = 0
    max_iter = 10000

    while True:
        iterations += 1
        x_next = [Fraction(0) for _ in range(n)]
        for i in range(n):
            s = sum(alpha[i][j] * x_curr[j] for j in range(n))
            x_next[i] = s + beta[i]
        
        diff = float(vector_norm_diff(x_next, x_curr))
        if diff <= eps or iterations >= max_iter:
            break
        x_curr = x_next

    return x_next, iterations, alpha_norm


def seidel_method(A, b, eps):
    """Метод Зейделя"""

    n = len(A)
    x_curr = [b[i] / A[i][i] for i in range(n)]

    iterations = 0
    max_iter = 10000

    while True:
        iterations += 1
        x_next = x_curr[:]

        for i in range(n):
            s1 = sum(A[i][j] * x_next[j] for j in range(i))
            s2 = sum(A[i][j] * x_curr[j] for j in range(i + 1, n))
            x_next[i] = (b[i] - s1 - s2) / A[i][i]

        diff = float(vector_norm_diff(x_next, x_curr))
        if diff <= eps or iterations >= max_iter:
            break
        x_curr = x_next

    return x_next, iterations


def main():

    filename = sys.argv[1] if len(sys.argv) > 1 else './1_3_tests/1.txt'
    n, A, b = read_input(filename)
    eps = 0.01

    print("==================================================")
    print("ЛАБОРАТОРНАЯ РАБОТА № 1.3: ИТЕРАЦИОННЫЕ МЕТОДЫ")
    print(f"Чтение данных из файла: {filename}")
    print("==================================================\n")

    print("--- Исходная система Ax = b ---")
    for i in range(n):
        row_str = " + ".join(f"({A[i][j]})*x_{j+1}" for j in range(n))
        print(f"  {row_str} = {b[i]}")
    print()

    L, U, P, _ = lu_decomposition(A)
    x_exact = solve_lu(L, U, P, b)

    x_simple, iter_simple, alpha_norm = simple_iteration_method(A, b, eps)
    
    x_seidel, iter_seidel = seidel_method(A, b, eps)

    print("==================================================")
    print(f"1. ЗАДАННА ТОЧНОСТЬ ВЫЧИСЛЕНИЙ ε = {eps}")
    print(f"   Норма матрицы alpha: ||alpha||_inf = {alpha_norm:.6f}")
    if alpha_norm < 1:
        print("   Достаточное условие сходимости ||alpha|| < 1 ВЫПОЛНЕНО!")
    else:
        print("   ВНИМАНИЕ: ||alpha|| >= 1 (сходимость не гарантируется в общем случае).")
    print("==================================================\n")

    print("2. КОЛИЧЕСТВО ИТЕРАЦИЙ:")
    print(f"  - Метод простых итераций (Якоби): {iter_simple} итераций")
    print(f"  - Метод Зейделя:                 {iter_seidel} итераций")
    print("==================================================\n")

    print("3. ЭТАЛОННОЕ ТОЧНОЕ РЕШЕНИЕ (Найденное через LU-разложение из 1.1):")
    print("  x* (дроби):    ", [str(v) for v in x_exact])
    print("  x* (десятичный):", [float(v) for v in x_exact])
    print("==================================================\n")

    print("   РЕЗУЛЬТАТЫ МЕТОДОВ:")
    print("   Метод простых итераций:")
    print("     x =", [float(v) for v in x_simple])
    print(f"     Погрешность ||x - x*|| = {float(vector_norm_diff(x_simple, x_exact)):.6e}")
    print()
    print("   Метод Зейделя:")
    print("     x =", [float(v) for v in x_seidel])
    print(f"     Погрешность ||x - x*|| = {float(vector_norm_diff(x_seidel, x_exact)):.6e}")
    print("==================================================\n")

    print("4. СРАВНЕНИЕ МЕТОДОВ ПО СКОРОСТИ СХОДИМОСТИ:")
    if iter_seidel < iter_simple:
        ratio = iter_simple / iter_seidel
        print(f"  [ВЫВОД] Метод Зейделя сошелся в {ratio:.2f} раз быстрей, чем метод простых итераций ({iter_seidel} итераций против {iter_simple}).")
    elif iter_seidel == iter_simple:
        print("  [ВЫВОД] Оба метода сошлись за одинаковое число итераций.")
    else:
        print("  [ВЫВОД] Метод простых итераций сошелся быстрее.")
    print("==================================================")


if __name__ == "__main__":
    main()