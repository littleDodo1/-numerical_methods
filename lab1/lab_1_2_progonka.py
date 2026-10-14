import os
import sys
from fractions import Fraction
from lab_1_1_LU import parse_number


def read_input_tridiagonal(filename):
    """Чтение матрицы из файла"""

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден!")

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError("Файл пуст!")

    n = int(lines[0])

    a = [Fraction(0)] * n
    b = [Fraction(0)] * n  
    c = [Fraction(0)] * n  

    for i in range(n):
        row = [parse_number(x) for x in lines[i + 1].split()]
        if len(row) != n:
            raise ValueError(f"Ошибка в {i+1}: ожидалось элементов {n}, получено {len(row)}")
        
        for j in range(n):
            if j not in (i - 1, i, i + 1) and row[j] != Fraction(0):
                raise ValueError("Метод неприменим: матрица не является трехдиагональной!")
               
        b[i] = row[i]

        if i > 0:
            a[i] = row[i - 1]

        if i < n - 1:
            c[i] = row[i + 1]

    d = [parse_number(x) for x in lines[n + 1].split()]
    if len(d) != n:
        raise ValueError(f"Ошибка в векторе d: ожидалось элементов {n}, получено {len(d)}")

    return n, a, b, c, d


def check_stability(a, b, c):
    """Проверка диагонального преобладания"""

    n = len(b)
    strict = False

    for i in range(n):
        sum_off = abs(a[i]) + abs(c[i])

        if abs(b[i]) < sum_off:
            return False, f"На строке {i+1}: |b_{i+1}| < |a_{i+1}| + |c_{i+1}|"
        
        if abs(b[i]) > sum_off:
            strict = True

    if not strict:
        return True, "Выполнено нестрогое диагональное преобладание (|b_i| >= |a_i| + |c_i|)"
    
    return True, "Выполнено строгое диагональное преобладание (|b_i| >= |a_i| + |c_i|)"


def solve_tridiagonal(a, b, c, d):
    """Метод прогонки (P, Q, x)"""

    n = len(b)
    P = [Fraction(0)] * n
    Q = [Fraction(0)] * n

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]

        if denom == 0:
            raise ValueError(f"Ошибка: Знаменатель равен нулю на шаге {i+1}")
        
        if i < n - 1:
            P[i] = -c[i] / denom
        else:
            P[i] = Fraction(0)
            
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    x = [Fraction(0)] * n
    x[n - 1] = Q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return P, Q, x


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "./1_2_tests/t_1.txt"
    print("==================================================")
    print("ЛАБОРАТОРНАЯ РАБОТА № 1.2: МЕТОД ПРОГОНКИ")
    print("==================================================\n")

    n, a, b, c, d = read_input_tridiagonal(filename)

    print(f"Размерность системы n = {n}")
    print("  Поддиагональ a:    [" + ", ".join(f"{str(x):>6}" for x in a) + "]")
    print("  Главная диагональ b:[" + ", ".join(f"{str(x):>6}" for x in b) + "]")
    print("  Наддиагональ c:    [" + ", ".join(f"{str(x):>6}" for x in c) + "]")
    print("  Правая часть d:    [" + ", ".join(f"{str(x):>6}" for x in d) + "]")
    print("\n" + "="*50 + "\n")

    is_stable, msg = check_stability(a, b, c)
    print("ПРОВЕРКА УСЛОВИЯ УСТОЙЧИВОСТИ ПРОГОНКИ:")
    print(f"  {msg}")
    print("\n" + "="*50 + "\n")

    P, Q, x = solve_tridiagonal(a, b, c, d)

    print("1. ПРОГОНОЧНЫЕ КОЭФФИЦИЕНТЫ:")
    print("  i |          P_i          |          Q_i          ")
    print("----+-----------------------+-----------------------")

    for i in range(n):
        print(f" {i+1:>2} | {str(P[i]):>21} | {str(Q[i]):>21}")
    print("\n" + "="*50 + "\n")

    print("2. ИТОГОВОЕ РЕШЕНИЕ СИСТЕМЫ УРАВНЕНИЙ x:")

    for i in range(n):
        print(f"  x_{i+1} = {str(x[i]):>12}  (десятичное: {float(x[i]):.10g})")
    print("\n" + "="*50 + "\n")

    print("ПРОВЕРКА ПОДСТАНОВКОЙ (A * x = d):")

    all_ok = True

    for i in range(n):
        left_val = a[i] * (x[i-1] if i > 0 else 0) + b[i] * x[i] + c[i] * (x[i+1] if i < n-1 else 0)
        diff = left_val - d[i]
        ok = (diff == 0)

        if not ok:
            all_ok = False
        print(f"  Уравнение {i+1}: левая часть = {left_val}, правая часть = {d[i]} -> OK: {ok}")

    print(f"\n[РЕЗУЛЬТАТ] Все уравнения удовлетворены точно: {all_ok}")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()