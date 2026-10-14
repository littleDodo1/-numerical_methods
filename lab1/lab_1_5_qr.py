import math
import numpy as np
import sys
import os

from lab_1_1_LU import matrix_mul
from lab_1_4_jacobi import read_input, format_matrix


def transpose(M):
    """Транспонирование 2D-матрицы"""
    n = len(M)
    m = len(M[0])
    return [[M[j][i] for j in range(n)] for i in range(m)]


def identity_matrix(n):
    """Единичная матрица размерности n x n"""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def vector_norm(v):
    """Евклидова норма вектора"""
    return math.sqrt(sum(x ** 2 for x in v))


def qr_decomposition_householder(A):
    """QR-разложение матрицы A"""

    n = len(A)
    R = [row[:] for row in A]
    Q = identity_matrix(n)

    for k in range(n - 1):
        x = [R[i][k] for i in range(k, n)]
        norm_x = vector_norm(x)

        if norm_x < 1e-15:
            continue

        sgn = 1.0 if x[0] >= 0 else -1.0
        v = x[:]
        v[0] += sgn * norm_x

        norm_v = vector_norm(v)
        if norm_v < 1e-15:
            continue

        v = [v_i / norm_v for v_i in v] 

        H_k = identity_matrix(n)
        m_sub = len(v)
        for i in range(m_sub):
            for j in range(m_sub):
                H_k[k + i][k + j] -= 2.0 * v[i] * v[j]

        R = matrix_mul(H_k, R)
        Q = matrix_mul(Q, H_k)

    return Q, R

def qr_algorithm(A, eps, max_iter=1000):
    """QR-алгоритм поиска собственных значений матрицы A"""
    
    n = len(A)
    A_k = [row[:] for row in A]
    
    for iteration in range(max_iter):
        Q_k, R_k = qr_decomposition_householder(A_k)
        A_next = matrix_mul(R_k, Q_k)

        diff = max(abs(A_next[i][j] - A_k[i][j]) for i in range(n) for j in range(n))
        if diff < eps:
            A_k = A_next
            break

        A_k = A_next

    eigenvalues = []
    i = 0

    while i < n:
        if i == n - 1:
            eigenvalues.append(complex(A_k[i][i], 0.0))
            i += 1
        else:
            if abs(A_k[i + 1][i]) < 1e-6:
                eigenvalues.append(complex(A_k[i][i], 0.0))
                i += 1
            else:
                a11, a12 = A_k[i][i], A_k[i][i + 1]
                a21, a22 = A_k[i + 1][i], A_k[i + 1][i + 1]

                trace = a11 + a22
                det = a11 * a22 - a12 * a21
                disc = trace ** 2 - 4.0 * det

                if disc >= 0:
                    l1 = (trace + math.sqrt(disc)) / 2.0
                    l2 = (trace - math.sqrt(disc)) / 2.0
                    eigenvalues.append(complex(l1, 0.0))
                    eigenvalues.append(complex(l2, 0.0))
                else:
                    real_part = trace / 2.0
                    imag_part = math.sqrt(-disc) / 2.0
                    eigenvalues.append(complex(real_part, imag_part))
                    eigenvalues.append(complex(real_part, -imag_part))
                i += 2

    return eigenvalues, A_k


def main():
    filename = sys.argv[1] if len(sys.argv) > 1 else "./1_5_tests/1.txt"
    n, A = read_input(filename)
    eps = 0.01

    print("==================================================")
    print("ЛАБОРАТОРНАЯ РАБОТА № 1.5 / 1.6: QR-РАЗЛОЖЕНИЕ И QR-АЛГОРИТМ")
    print(f"Чтение данных из файла: {filename}")
    print("==================================================\n")

    print(format_matrix(A, "1. Исходная матрица A"))
    print("\n" + "="*50 + "\n")

    Q, R = qr_decomposition_householder(A)

    print(format_matrix(Q, "2. Ортогональная матрица Q"))
    print()
    print(format_matrix(R, "   Верхняя треугольная матрица R"))
    print("\n" + "="*50 + "\n")

    QR = matrix_mul(Q, R)
    print(format_matrix(QR, "3. Результат умножения Q · R"))
    
    max_diff = max(abs(QR[i][j] - A[i][j]) for i in range(n) for j in range(n))
    print(f"\n[ПРОВЕРКА] Максимальное отклонение ||Q·R - A|| = {max_diff:.6e}")
    print(f"   Совпадение Q · R = A с точностью ε: {max_diff <= eps * 10}\n")
    print("="*50 + "\n")

    # 4. QR-алгоритм поиска собственных значений
    eigenvalues, A_final = qr_algorithm(A, eps)

    print("4. НАЙДЕННЫЕ СОБСТВЕННЫЕ ЗНАЧЕНИЯ (через QR-алгоритм):")
    for idx, lam in enumerate(eigenvalues, 1):
        if abs(lam.imag) < 1e-8:
            print(f"   λ_{idx} = {lam.real:12.6f}")
        else:
            sgn = "+" if lam.imag >= 0 else "-"
            print(f"   λ_{idx} = {lam.real:12.6f} {sgn} {abs(lam.imag):12.6f}i")

    print("\n" + "="*50 + "\n")

    # Проверка через NumPy
    try:
        np_A = np.array(A, dtype=float)
        np_eigs = np.linalg.eigvals(np_A)
        print("--- Проверка через NumPy (np.linalg.eigvals) ---")
        for idx, lam in enumerate(np_eigs, 1):
            if abs(lam.imag) < 1e-8:
                print(f"   NumPy λ_{idx} = {lam.real:12.6f}")
            else:
                sgn = "+" if lam.imag >= 0 else "-"
                print(f"   NumPy λ_{idx} = {lam.real:12.6f} {sgn} {abs(lam.imag):12.6f}i")
    except ImportError:
        pass

    print("==================================================")


if __name__ == "__main__":
    main()
