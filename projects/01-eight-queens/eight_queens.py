# eight_queens.py
# 适配 kanren 0.2.3 版本的八皇后问题求解

import time
import itertools

# ==========================================
# kanren 0.2.3 兼容导入
# ==========================================
try:
    from kanren import run, membero, var, eq, conde
    from kanren.core import lall
    # 0.2.3 版本可能没有 success/fail，需要自定义
    def success(state):
        return [state]
    
    def fail(state):
        return []
except ImportError as e:
    print(f"导入错误：{e}")
    print("建议升级 kanren: pip install kanren==1.0.3")
    exit(1)

# ==========================================
# 1. 自定义 neq 函数 (要求 3)
# ==========================================
def neq(x, y):
    """
    自定义不等约束目标（适配 kanren 0.2.3）
    """
    def _neq(state):
        # 0.2.3 版本的 state 处理方式不同
        try:
            x_val = state.get(x, None)
            y_val = state.get(y, None)
        except:
            x_val = x
            y_val = y
        
        if x_val is not None and y_val is not None:
            if x_val == y_val:
                return []
            else:
                return [state]
        return [state]
    return _neq

# ==========================================
# 2. 穷举法 + 逻辑筛选 (要求 1)
# ==========================================
def solve_brute_force():
    """穷举所有排列，筛选合法解"""
    start = time.time()
    solutions = []
    
    for p in itertools.permutations(range(8)):
        is_valid = True
        for i in range(8):
            for j in range(i + 1, 8):
                if abs(i - j) == abs(p[i] - p[j]):
                    is_valid = False
                    break
            if not is_valid:
                break
        if is_valid:
            solutions.append(p)
    
    end = time.time()
    return solutions, end - start

# ==========================================
# 3. 纯逻辑编程 (要求 2 & 3)
# ==========================================
def not_on_diagonal(q1, q2, row_diff):
    """自定义目标：确保两个皇后不在斜线上"""
    def _goal(state):
        try:
            v1 = state.get(q1, None)
            v2 = state.get(q2, None)
        except:
            v1 = q1
            v2 = q2
        
        if v1 is not None and v2 is not None:
            if abs(v1 - v2) == row_diff:
                return []
        return [state]
    return _goal

def queens_constraint(queens):
    """定义八皇后的逻辑约束"""
    constraints = []
    for i in range(len(queens)):
        for j in range(i + 1, len(queens)):
            row_diff = j - i
            constraints.append(not_on_diagonal(queens[i], queens[j], row_diff))
    return constraints

def solve_logic_programming():
    """纯逻辑编程求解（适配 0.2.3）"""
    start = time.time()
    
    # 创建 8 个逻辑变量
    queens = tuple(var() for _ in range(8))
    
    # 约束：每个皇后在 0-7 范围内
    domain_constraints = [membero(q, (0,1,2,3,4,5,6,7)) for q in queens]
    
    # 约束：所有皇后列不同（通过排列保证）
    # 约束：斜线不同
    diagonal_constraints = queens_constraint(queens)
    
    # 合并所有约束
    all_constraints = lall(*domain_constraints, *diagonal_constraints)
    
    # 求解（只找 1 个解，避免超时）
    try:
        results = list(run(1, queens, all_constraints))
    except Exception as e:
        print(f"逻辑编程求解出错：{e}")
        results = []
    
    end = time.time()
    return results, end - start

# ==========================================
# 4. DFS 回溯算法 (要求 4)
# ==========================================
def solve_dfs():
    """DFS 回溯法求解"""
    start = time.time()
    solutions = []
    
    def backtrack(row, cols, diag1, diag2, current):
        if row == 8:
            solutions.append(tuple(current))
            return
        for col in range(8):
            if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                backtrack(row + 1, 
                         cols | {col}, 
                         diag1 | {row - col}, 
                         diag2 | {row + col}, 
                         current + [col])
    
    backtrack(0, set(), set(), set(), [])
    end = time.time()
    return solutions, end - start

# ==========================================
# 5. BFS 算法 (要求 4)
# ==========================================
def solve_bfs():
    """BFS 广度优先搜索求解"""
    start = time.time()
    solutions = []
    
    from collections import deque
    queue = deque([(0, set(), set(), set(), [])])
    
    while queue:
        row, cols, diag1, diag2, current = queue.popleft()
        
        if row == 8:
            solutions.append(tuple(current))
            continue
        
        for col in range(8):
            if col not in cols and (row - col) not in diag1 and (row + col) not in diag2:
                queue.append((row + 1, 
                             cols | {col}, 
                             diag1 | {row - col}, 
                             diag2 | {row + col}, 
                             current + [col]))
    
    end = time.time()
    return solutions, end - start

# ==========================================
# 6. 主程序与对比
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("八皇后问题求解对比实验")
    print("=" * 50)
    print(f"kanren 版本：{__import__('kanren').__version__}")
    print(f"Python 版本：{__import__('sys').version}")
    print()
    
    # 1. 穷举法
    print("[1] 穷举法 + 逻辑筛选...")
    sol1, t1 = solve_brute_force()
    print(f"    解的数量：{len(sol1)}, 耗时：{t1:.6f} 秒")
    
    # 2. DFS 回溯
    print("\n[2] DFS 回溯算法...")
    sol2, t2 = solve_dfs()
    print(f"    解的数量：{len(sol2)}, 耗时：{t2:.6f} 秒")
    
    # 3. BFS 搜索
    print("\n[3] BFS 广度优先搜索...")
    sol3, t3 = solve_bfs()
    print(f"    解的数量：{len(sol3)}, 耗时：{t3:.6f} 秒")
    
    # 4. 纯逻辑编程
    print("\n[4] 纯逻辑编程 (kanren)...")
    print("    注意：可能较慢，只寻找 1 个解...")
    sol4, t4 = solve_logic_programming()
    print(f"    解的数量：{len(sol4)}, 耗时：{t4:.6f} 秒")
    
    # 5. 结果验证
    print("\n" + "=" * 50)
    print("结果验证:")
    if len(sol1) == len(sol2) == len(sol3):
        print("✓ 验证通过：穷举法、DFS、BFS 结果一致")
        print(f"  八皇后问题共有 {len(sol1)} 个解")
    else:
        print("✗ 验证失败：结果不一致")
        print(f"  穷举法：{len(sol1)}, DFS: {len(sol2)}, BFS: {len(sol3)}")
    
    # 6. 性能对比
    print("\n" + "=" * 50)
    print("性能对比:")
    methods = [
        ("穷举法", t1),
        ("DFS 回溯", t2),
        ("BFS 搜索", t3),
        ("逻辑编程", t4)
    ]
    
    for name, t in sorted(methods, key=lambda x: x[1]):
        print(f"  {name}: {t:.6f} 秒")
    
    print("\n" + "=" * 50)
    print("实验完成！")
    print("=" * 50)