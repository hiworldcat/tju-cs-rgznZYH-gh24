"""
Robot Obstacle Avoidance Path Planning System
Implemented Algorithms: BFS, A*, Dijkstra, Genetic Algorithm (GA), Artificial Potential Field (APF)
Includes dynamic visualization and performance comparison analysis
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import heapq
import time
from typing import List, Tuple, Dict, Set
import random
import warnings
from matplotlib.animation import FuncAnimation, PillowWriter
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']

class RobotPathPlanning:
    def __init__(self, grid, start, goal):
        """
        Initialize path planning system
        
        Parameters:
        -----------
        grid : np.ndarray
            Grid map, 0 represents traversable, 1 represents obstacle
        start : tuple
            Start position (row, col)
        goal : tuple
            Goal position (row, col)
        """
        self.grid = grid.copy()
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape
        
        # Define move directions: up, down, left, right, up-right, up-left, down-right, down-left
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]
        self.direction_cost = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
        
        # Color map for visualization
        self.cmap = ListedColormap(['white', 'black', 'lightblue', 'yellow', 'green', 'red', 'orange'])
        
        # 创建算法对应的文件夹
        self.algorithm_folders = {
            'BFS': 'results/BFS',
            'A*': 'results/A_star',
            'Dijkstra': 'results/Dijkstra',
            'GA': 'results/GA',
            'APF': 'results/APF'
        }
        
        # 创建所有需要的文件夹
        for folder in self.algorithm_folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
        
        # 存储搜索历史用于制作GIF
        self.search_history = {}
        
    def is_valid(self, pos):
        """Check if position is valid (within map and not obstacle)"""
        row, col = pos
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row, col] == 0
        return False
    
    def heuristic(self, pos):
        """Heuristic function: combination of Manhattan distance and Euclidean distance"""
        manhattan = abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
        euclidean = np.sqrt((pos[0] - self.goal[0])**2 + (pos[1] - self.goal[1])**2)
        return manhattan + 0.5 * euclidean
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def save_single_image(self, explored_nodes, path, title, algorithm_name, suffix=""):
        """保存单张图片（用于最终结果）- 修复颜色映射问题"""
        plt.figure(figsize=(10, 8))
        
        # 创建显示网格，使用整数类型
        display_grid = np.zeros_like(self.grid, dtype=int)
        
        # 先设置障碍物（优先级最高）
        display_grid[self.grid == 1] = 1
        
        # 标记探索过的节点
        for node in explored_nodes:
            if node is not None and 0 <= node[0] < self.rows and 0 <= node[1] < self.cols:
                if display_grid[node[0], node[1]] == 0:  # 只覆盖空地
                    display_grid[node[0], node[1]] = 2
        
        # 标记路径（优先级高于探索节点）
        if path:
            for node in path:
                if node is not None and 0 <= node[0] < self.rows and 0 <= node[1] < self.cols:
                    display_grid[node[0], node[1]] = 3
        
        # 标记起点和终点（最高优先级）
        if 0 <= self.start[0] < self.rows and 0 <= self.start[1] < self.cols:
            display_grid[self.start[0], self.start[1]] = 4
        if 0 <= self.goal[0] < self.rows and 0 <= self.goal[1] < self.cols:
            display_grid[self.goal[0], self.goal[1]] = 5
        
        plt.imshow(display_grid, cmap=self.cmap, interpolation='nearest', vmin=0, vmax=6)
        plt.title(title)
        
        # 确保颜色条按照正确的标签显示
        cbar = plt.colorbar(ticks=[0, 1, 2, 3, 4, 5])
        cbar.ax.set_yticklabels(['Land', 'Obstacle', 'Explored', 'Path', 'Start', 'Goal'])
        
        # 确定保存路径
        safe_title = title.replace(' ', '_').replace(':', '').replace('/', '_')
        folder = self.algorithm_folders.get(algorithm_name, 'results/others')
        
        if suffix:
            filename = f"{safe_title}_{suffix}.png"
        else:
            filename = f"{safe_title}.png"
        filepath = os.path.join(folder, filename)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")
    
    def create_animation(self, frames_data, algorithm_name, title):
        """创建动画GIF - 修复颜色映射问题"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update_frame(frame_idx):
            ax.clear()
            explored_nodes, path = frames_data[frame_idx]
            
            # 创建显示网格，使用整数类型避免浮点数精度问题
            display_grid = np.zeros_like(self.grid, dtype=int)
            
            # 先设置障碍物（优先级最高）
            display_grid[self.grid == 1] = 1
            
            # 标记探索过的节点
            for node in explored_nodes:
                if node is not None and 0 <= node[0] < self.rows and 0 <= node[1] < self.cols:
                    if display_grid[node[0], node[1]] == 0:  # 只覆盖空地
                        display_grid[node[0], node[1]] = 2
            
            # 标记路径（优先级高于探索节点）
            if path:
                for node in path:
                    if node is not None and 0 <= node[0] < self.rows and 0 <= node[1] < self.cols:
                        display_grid[node[0], node[1]] = 3
            
            # 标记起点和终点（最高优先级）
            if 0 <= self.start[0] < self.rows and 0 <= self.start[1] < self.cols:
                display_grid[self.start[0], self.start[1]] = 4
            if 0 <= self.goal[0] < self.rows and 0 <= self.goal[1] < self.cols:
                display_grid[self.goal[0], self.goal[1]] = 5
            
            # 使用matplotlib的显式颜色映射
            im = ax.imshow(display_grid, cmap=self.cmap, interpolation='nearest', vmin=0, vmax=6)
            ax.set_title(f"{title} - Step {frame_idx + 1}/{len(frames_data)}")
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 创建动画
        anim = FuncAnimation(fig, update_frame, frames=len(frames_data), 
                        interval=100, repeat=False)
        
        # 保存为GIF
        gif_path = os.path.join(self.algorithm_folders[algorithm_name], 
                            f"{title.replace(' ', '_')}.gif")
        anim.save(gif_path, writer=PillowWriter(fps=10))
        plt.close()
        print(f"Animation saved: {gif_path}")
    
    def bfs(self, visualize=True):
        """
        BFS algorithm implementation
        Principle: Breadth-first search using queue, guarantees shortest path (unweighted graph)
        """
        print("\n=== Running BFS Algorithm ===")
        start_time = time.time()
        
        queue = deque([self.start])
        came_from = {self.start: None}
        visited = set([self.start])
        explored_nodes = []
        algorithm_name = "BFS"
        
        # 存储关键帧 - 提高频率以获得更流畅的动画（原来每10%保存一次，现在每5%保存一次）
        frames = []
        last_frame_count = 0
        total_nodes_estimate = self.rows * self.cols
        
        while queue:
            current = queue.popleft()
            explored_nodes.append(current)
            
            # 保存关键帧：提高帧率，每探索约5%的节点保存一次（原来是10%）
            current_count = len(explored_nodes)
            if visualize and (current_count - last_frame_count) >= 5:
                frames.append((explored_nodes.copy(), []))
                last_frame_count = current_count
            
            if current == self.goal:
                path = self.reconstruct_path(came_from, current)
                elapsed_time = time.time() - start_time
                
                print(f"BFS found path!")
                print(f"Path length: {len(path)}")
                print(f"Explored nodes: {len(explored_nodes)}")
                print(f"Run time: {elapsed_time:.4f} seconds")
                
                if visualize:
                    # 添加最终路径帧
                    frames.append((explored_nodes, path))
                    # 创建动画
                    if len(frames) > 1:
                        self.create_animation(frames, algorithm_name, "BFS_Search_Process")
                    # 保存最终结果图片
                    self.save_single_image(explored_nodes, path, 
                                         "BFS Final Path", algorithm_name, "final")
                
                return path, explored_nodes, elapsed_time
            
            # Explore neighbors
            for i, (dx, dy) in enumerate(self.directions):
                neighbor = (current[0] + dx, current[1] + dy)
                if self.is_valid(neighbor) and neighbor not in visited:
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        print("BFS did not find a path!")
        return None, explored_nodes, time.time() - start_time
    
    def astar(self, visualize=True):
        """
        A* algorithm implementation
        Principle: Uses heuristic function f(n) = g(n) + h(n), prioritizes most promising nodes
        """
        print("\n=== Running A* Algorithm ===")
        start_time = time.time()
        
        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set = [(self.heuristic(self.start), counter, self.start)]
        
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start)}
        
        explored_nodes = []
        visited = set([self.start])
        algorithm_name = "A*"
        
        # 存储关键帧 - 提高频率
        frames = []
        last_frame_count = 0
        total_nodes_estimate = self.rows * self.cols
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            explored_nodes.append(current)
            
            # 保存关键帧：提高帧率，每探索约5%的节点保存一次
            current_count = len(explored_nodes)
            if visualize and (current_count - last_frame_count) >= 5:
                frames.append((explored_nodes.copy(), []))
                last_frame_count = current_count
            
            if current == self.goal:
                path = self.reconstruct_path(came_from, current)
                elapsed_time = time.time() - start_time
                
                print(f"A* found path!")
                print(f"Path length: {len(path)}")
                print(f"Explored nodes: {len(explored_nodes)}")
                print(f"Run time: {elapsed_time:.4f} seconds")
                
                if visualize:
                    # 添加最终路径帧
                    frames.append((explored_nodes, path))
                    # 创建动画
                    if len(frames) > 1:
                        self.create_animation(frames, algorithm_name, "A_star_Search_Process")
                    # 保存最终结果图片
                    self.save_single_image(explored_nodes, path, 
                                         "A* Final Path", algorithm_name, "final")
                
                return path, explored_nodes, elapsed_time
            
            # Explore neighbors
            for i, (dx, dy) in enumerate(self.directions):
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid(neighbor):
                    continue
                
                tentative_g = g_score[current] + self.direction_cost[i]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
                    visited.add(neighbor)
        
        print("A* did not find a path!")
        return None, explored_nodes, time.time() - start_time
    
    def dijkstra(self, visualize=True):
        """
        Dijkstra algorithm implementation
        Principle: Similar to BFS but uses priority queue to handle edges with different costs
        """
        print("\n=== Running Dijkstra Algorithm ===")
        start_time = time.time()
        
        # Priority queue: (cost, position)
        open_set = [(0, self.start)]
        came_from = {}
        cost_so_far = {self.start: 0}
        explored_nodes = []
        algorithm_name = "Dijkstra"
        
        # 存储关键帧 - 提高频率
        frames = []
        last_frame_count = 0
        total_nodes_estimate = self.rows * self.cols
        
        while open_set:
            current_cost, current = heapq.heappop(open_set)
            explored_nodes.append(current)
            
            # 保存关键帧：提高帧率，每探索约5%的节点保存一次
            current_count = len(explored_nodes)
            if visualize and (current_count - last_frame_count) >= 5:
                frames.append((explored_nodes.copy(), []))
                last_frame_count = current_count
            
            if current == self.goal:
                path = self.reconstruct_path(came_from, current)
                elapsed_time = time.time() - start_time
                
                print(f"Dijkstra found path!")
                print(f"Path length: {len(path)}")
                print(f"Explored nodes: {len(explored_nodes)}")
                print(f"Run time: {elapsed_time:.4f} seconds")
                
                if visualize:
                    # 添加最终路径帧
                    frames.append((explored_nodes, path))
                    # 创建动画
                    if len(frames) > 1:
                        self.create_animation(frames, algorithm_name, "Dijkstra_Search_Process")
                    # 保存最终结果图片
                    self.save_single_image(explored_nodes, path, 
                                         "Dijkstra Final Path", algorithm_name, "final")
                
                return path, explored_nodes, elapsed_time
            
            # Explore neighbors
            for i, (dx, dy) in enumerate(self.directions):
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid(neighbor):
                    continue
                
                new_cost = current_cost + self.direction_cost[i]
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (new_cost, neighbor))
        
        print("Dijkstra did not find a path!")
        return None, explored_nodes, time.time() - start_time
    
    def genetic_algorithm(self, population_size=200, generations=500, visualize=True):
        """
        Genetic algorithm implementation for path planning
        Principle: Simulates natural selection and genetic variation to evolve optimal paths
        """
        print("\n=== Running Genetic Algorithm ===")
        start_time = time.time()
        algorithm_name = "GA"
        
        # 内部辅助函数
        def _create_greedy_path():
            """Create a greedy path to the goal"""
            path = [self.start]
            current = self.start
            visited = {self.start}
            
            for _ in range(self.rows * self.cols):
                if current == self.goal:
                    return path
                
                # 获取有效邻居
                neighbors = []
                for dx, dy in self.directions[:4]:  # 使用4方向
                    neighbor = (current[0] + dx, current[1] + dy)
                    if self.is_valid(neighbor) and neighbor not in visited:
                        neighbors.append(neighbor)
                
                if not neighbors:
                    break
                
                # 选择最接近目标的邻居
                distances = [abs(n[0] - self.goal[0]) + abs(n[1] - self.goal[1]) for n in neighbors]
                best_idx = np.argmin(distances)
                next_pos = neighbors[best_idx]
                
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
            
            return path if path[-1] == self.goal else None
        
        def _optimize_path(path):
            """Optimize path by removing redundant nodes"""
            if not path or len(path) < 3:
                return path
            
            # 第一步：移除可以直接跳过的节点
            optimized = [path[0]]
            i = 1
            
            while i < len(path) - 1:
                # 检查是否可以直接从最后一个优化节点到 path[i+1]
                if self._are_adjacent(optimized[-1], path[i+1]):
                    i += 1  # 跳过当前节点
                else:
                    optimized.append(path[i])
                    i += 1
            
            optimized.append(path[-1])
            
            # 第二步：移除直线上的冗余点
            i = 1
            while i < len(optimized) - 1:
                prev = optimized[i-1]
                curr = optimized[i]
                next_pos = optimized[i+1]
                
                # 检查三点是否共线（同一行或同一列）
                if (prev[0] == curr[0] == next_pos[0] or 
                    prev[1] == curr[1] == next_pos[1]):
                    optimized.pop(i)
                else:
                    i += 1
            
            return optimized
        
        def create_individual():
            """Create random individual (path) - improved version"""
            path = [self.start]
            current = self.start
            max_steps = self.rows * self.cols * 2
            visited = set([self.start])
            
            for _ in range(max_steps):
                if current == self.goal:
                    break
                
                # Get valid neighbors
                valid_moves = []
                for i, (dx, dy) in enumerate(self.directions):
                    neighbor = (current[0] + dx, current[1] + dy)
                    if self.is_valid(neighbor) and neighbor not in visited:
                        valid_moves.append(neighbor)
                
                if not valid_moves:
                    # Backtrack to find alternative path
                    if len(path) > 1:
                        path.pop()
                        current = path[-1] if path else self.start
                        continue
                    else:
                        break
                
                # Bias towards goal (with some randomness)
                if random.random() < 0.7:  # 70% greedy, 30% random
                    # Choose move that gets closer to goal
                    distances = []
                    for move in valid_moves:
                        dist = abs(move[0] - self.goal[0]) + abs(move[1] - self.goal[1])
                        distances.append(dist)
                    best_idx = np.argmin(distances)
                    next_pos = valid_moves[best_idx]
                else:
                    next_pos = random.choice(valid_moves)
                
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
            
            # Ensure path ends at goal (if possible)
            if current != self.goal:
                # Try to find a path from current to goal
                remaining_path = self._find_shortcut_to_goal(current)
                if remaining_path:
                    path.extend(remaining_path[1:])
            
            return path
        
        def fitness(path):
            """Improved fitness function"""
            if not path or len(path) < 2:
                return float('inf')
            
            # Path length penalty
            length_penalty = len(path)
            
            # Collision penalty (should be 0 for valid paths)
            collision_penalty = 0
            for pos in path:
                if not self.is_valid(pos):
                    collision_penalty += 1000
            
            # Distance to goal (final position)
            last_pos = path[-1]
            distance_to_goal = abs(last_pos[0] - self.goal[0]) + abs(last_pos[1] - self.goal[1])
            
            # Path smoothness penalty (reward straight lines)
            smoothness_penalty = 0
            for i in range(1, len(path)-1):
                prev = path[i-1]
                curr = path[i]
                next_pos = path[i+1]
                # Check if path is not straight
                if (prev[0] != next_pos[0] and prev[1] != next_pos[1]):
                    smoothness_penalty += 0.5
            
            # Goal achievement bonus
            goal_bonus = 0
            if last_pos == self.goal:
                goal_bonus = -500  # Large negative penalty (good)
            
            # Duplicate penalty (avoid loops)
            duplicate_penalty = (len(path) - len(set(path))) * 10
            
            return (length_penalty + collision_penalty + 
                    distance_to_goal * 5 + smoothness_penalty + 
                    duplicate_penalty + goal_bonus)
        
        def crossover(parent1, parent2):
            """Improved crossover operation"""
            if len(parent1) < 2 or len(parent2) < 2:
                return parent1[:]
            
            # Find common points (excluding start and goal if they're common)
            common_points = []
            for i, p1 in enumerate(parent1[1:-1]):  # Exclude start
                if p1 in parent2[1:-1] and p1 != self.start:
                    common_points.append((p1, i+1))  # Store position in parent1
            
            if not common_points:
                # No common points, use single-point crossover at random positions
                pos1 = random.randint(1, len(parent1)-1)
                pos2 = random.randint(1, len(parent2)-1)
                
                # Try to connect the two parts
                child = parent1[:pos1]
                remaining = parent2[pos2:]
                
                # Find a connection point
                if child and remaining:
                    last_child = child[-1]
                    # Check if last_child connects to first of remaining
                    if self._are_adjacent(last_child, remaining[0]):
                        child.extend(remaining)
                    else:
                        # Try to find a path between them
                        connector = self._find_shortcut(last_child, remaining[0])
                        if connector:
                            child.extend(connector[1:])
                            child.extend(remaining)
                        else:
                            return parent1[:]
                return child
            
            # Select random common point
            cross_point, idx1 = random.choice(common_points)
            idx2 = parent2.index(cross_point)
            
            # Create child
            child = parent1[:idx1] + parent2[idx2+1:] if idx2+1 < len(parent2) else parent1[:idx1]
            
            # Remove duplicates (keep first occurrence)
            seen = set()
            unique_child = []
            for pos in child:
                if pos not in seen:
                    seen.add(pos)
                    unique_child.append(pos)
            
            return unique_child
        
        def mutate(path, mutation_rate=0.15):
            """Improved mutation operation"""
            if random.random() > mutation_rate or len(path) < 3:
                return path
            
            mutated = path.copy()
            
            # Random mutation type
            mutation_type = random.choice(['replace', 'insert', 'delete', 'smooth'])
            
            if mutation_type == 'replace' and len(mutated) > 2:
                # Replace a random node (not start or goal)
                idx = random.randint(1, len(mutated)-2)
                current = mutated[idx]
                
                # Find alternative path around this node
                neighbors = []
                for dx, dy in self.directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if self.is_valid(neighbor):
                        neighbors.append(neighbor)
                
                if neighbors:
                    # Try to create a smoother path
                    prev = mutated[idx-1]
                    next_pos = mutated[idx+1]
                    
                    # Find a better connection between prev and next
                    shortcut = self._find_shortcut(prev, next_pos)
                    if shortcut and len(shortcut) < 4:
                        # Replace segment with shortcut
                        mutated = mutated[:idx] + shortcut[1:-1] + mutated[idx+1:]
            
            elif mutation_type == 'insert' and len(mutated) < self.rows * self.cols:
                # Insert a new node
                idx = random.randint(1, len(mutated)-1)
                current = mutated[idx-1]
                next_pos = mutated[idx]
                
                # Find intermediate point
                intermediate = self._find_midpoint(current, next_pos)
                if intermediate and intermediate != current and intermediate != next_pos:
                    mutated.insert(idx, intermediate)
            
            elif mutation_type == 'delete' and len(mutated) > 3:
                # Delete a redundant node
                idx = random.randint(1, len(mutated)-2)
                # Check if deletion maintains connectivity
                if self._are_adjacent(mutated[idx-1], mutated[idx+1]):
                    mutated.pop(idx)
            
            elif mutation_type == 'smooth' and len(mutated) > 3:
                # Smooth the path by removing intermediate points on straight lines
                i = 1
                while i < len(mutated) - 1:
                    prev = mutated[i-1]
                    curr = mutated[i]
                    next_pos = mutated[i+1]
                    
                    # Check if points are collinear (same row or same column)
                    if (prev[0] == curr[0] == next_pos[0] or 
                        prev[1] == curr[1] == next_pos[1]):
                        mutated.pop(i)
                    else:
                        i += 1
            
            # Ensure path validity
            valid_path = [mutated[0]]
            for i in range(1, len(mutated)):
                if self._are_adjacent(valid_path[-1], mutated[i]) and self.is_valid(mutated[i]):
                    valid_path.append(mutated[i])
                else:
                    # Try to connect the gap
                    connector = self._find_shortcut(valid_path[-1], mutated[i])
                    if connector:
                        valid_path.extend(connector[1:])
                    else:
                        break
            
            return valid_path if valid_path[-1] == self.goal else path
        
        # Initialize population
        print("Initializing population...")
        population = []
        for _ in range(population_size):
            individual = create_individual()
            if individual and individual[-1] == self.goal:
                population.append(individual)
            else:
                # Add a simple greedy path as fallback
                greedy_path = _create_greedy_path()  # 使用内部函数
                if greedy_path:
                    population.append(greedy_path)
                else:
                    population.append([self.start, self.goal])
        
        # Ensure we have enough valid individuals
        population = [ind for ind in population if ind and len(ind) >= 2]
        while len(population) < population_size:
            population.append([self.start, self.goal])
        
        best_path = None
        best_fitness = float('inf')
        no_improvement_count = 0
        
        # Store evolution history
        evolution_history = []
        last_evolution_save = 0
        
        print(f"Starting evolution with population size {population_size}")
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [fitness(ind) for ind in population]
            
            # Update best solution
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_path = population[min_fitness_idx].copy()
                no_improvement_count = 0
                
                # Save evolution history for visualization
                if visualize and generation - last_evolution_save >= 20:
                    evolution_history.append((generation, best_path.copy()))
                    last_evolution_save = generation
                    print(f"Generation {generation}: Best fitness = {best_fitness:.2f}, Path length = {len(best_path)}")
            else:
                no_improvement_count += 1
            
            # Early termination if converged
            if no_improvement_count > 50 and best_path and best_path[-1] == self.goal:
                print(f"GA converged at generation {generation}")
                break
            
            # Selection (tournament selection)
            selected = []
            elite_count = max(2, population_size // 10)  # Elitism
            
            # Keep elite individuals
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            for idx in elite_indices:
                selected.append(population[idx])
            
            # Tournament selection for the rest
            while len(selected) < population_size:
                tournament_size = 5
                tournament_indices = random.sample(range(len(population)), 
                                                min(tournament_size, len(population)))
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                selected.append(population[winner_idx])
            
            # Crossover and mutation
            new_population = []
            
            # Keep elites
            for i in range(elite_count):
                new_population.append(selected[i])
            
            # Generate offspring
            while len(new_population) < population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Crossover with probability
                if random.random() < 0.8:
                    child = crossover(parent1, parent2)
                else:
                    child = parent1[:]
                
                # Mutation
                child = mutate(child, mutation_rate=0.1)
                
                # Validate child
                if child and len(child) >= 2 and self.is_valid(child[-1]):
                    new_population.append(child)
                else:
                    # Add a random individual as fallback
                    new_population.append(random.choice(selected))
            
            population = new_population
            
            # Progress report
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
        
        elapsed_time = time.time() - start_time
        
        # Final validation and optimization
        if best_path and best_path[-1] == self.goal:
            # Remove any cycles or redundant nodes
            final_path = _optimize_path(best_path)  # 使用内部函数
            
            print(f"GA found path!")
            print(f"Original path length: {len(best_path)}")
            print(f"Optimized path length: {len(final_path)}")
            print(f"Generations: {generation + 1}")
            print(f"Run time: {elapsed_time:.4f} seconds")
            
            if visualize:
                # Create GA evolution animation
                if evolution_history:
                    print(f"Creating evolution animation with {len(evolution_history)} frames...")
                    frames = [( [], path) for _, path in evolution_history]
                    self.create_animation(frames, algorithm_name, "GA_Evolution_Process")
                # Save final result
                self.save_single_image([], final_path, 
                                    "GA Final Path", algorithm_name, "final")
            
            return final_path, None, elapsed_time
        else:
            print("GA did not find a valid path!")
            # Create a basic path as fallback
            fallback_path = _create_greedy_path()  # 使用内部函数
            if fallback_path:
                return fallback_path, None, elapsed_time
            return None, None, elapsed_time

    def artificial_potential_field(self, max_iter=5000, visualize=True):
        """
        Artificial Potential Field method
        Principle: Goal generates attractive field, obstacles generate repulsive field, resultant force guides robot motion
        """
        print("\n=== Running Artificial Potential Field Method ===")
        start_time = time.time()
        algorithm_name = "APF"
        
        # Parameter settings
        k_att = 1.0   # 减小吸引力，避免震荡
        k_rep = 100.0  # 减小斥力
        d0 = 5.0      # 增大影响距离
        step_size = 0.5  # 减小步长，更精细
        # 添加随机扰动逃逸局部极小值
        
        def attractive_force(pos):
            """Calculate attractive force"""
            direction = (self.goal[0] - pos[0], self.goal[1] - pos[1])
            distance = np.sqrt(direction[0]**2 + direction[1]**2)
            if distance > 0:
                force_magnitude = k_att * distance
                return (force_magnitude * direction[0] / distance,
                    force_magnitude * direction[1] / distance)
            return (0, 0)
        
        def repulsive_force(pos):
            """Calculate repulsive force"""
            force = (0, 0)
            
            # Check all obstacles
            for i in range(self.rows):
                for j in range(self.cols):
                    if self.grid[i, j] == 1:  # Obstacle
                        obstacle_pos = (i, j)
                        direction = (pos[0] - obstacle_pos[0], pos[1] - obstacle_pos[1])
                        distance = np.sqrt(direction[0]**2 + direction[1]**2)
                        
                        if distance < d0 and distance > 0:
                            force_magnitude = k_rep * (1/distance - 1/d0) / (distance**2)
                            force = (force[0] + force_magnitude * direction[0] / distance,
                                    force[1] + force_magnitude * direction[1] / distance)
            
            return force
        
        # Path planning
        current_pos = list(self.start)
        path = [tuple(current_pos)]
        explored_nodes = [tuple(current_pos)]
        
        # 存储关键帧 - 提高频率，每25次迭代保存一次（原来是50次）
        frames = []
        last_frame_count = 0
        
        for iteration in range(max_iter):
            # Calculate resultant force
            f_att = attractive_force(tuple(current_pos))
            f_rep = repulsive_force(tuple(current_pos))
            f_total = (f_att[0] + f_rep[0], f_att[1] + f_rep[1])
            
            # Update position
            f_norm = np.sqrt(f_total[0]**2 + f_total[1]**2)
            if f_norm > 0:
                step_vec = (step_size * f_total[0] / f_norm,
                        step_size * f_total[1] / f_norm)
                current_pos[0] += step_vec[0]
                current_pos[1] += step_vec[1]
            
            # Round to nearest grid cell
            grid_pos = (int(round(current_pos[0])), int(round(current_pos[1])))
            
            # Boundary check
            if not (0 <= grid_pos[0] < self.rows and 0 <= grid_pos[1] < self.cols):
                break
            
            # Collision check
            if self.grid[grid_pos[0], grid_pos[1]] == 1:
                break
            
            if grid_pos not in path:
                path.append(grid_pos)
                explored_nodes.append(grid_pos)
            
            # 保存关键帧：提高帧率，每25次迭代保存一次（原来是50次）
            if visualize and (iteration - last_frame_count) > 25:
                frames.append((explored_nodes.copy(), path.copy()))
                last_frame_count = iteration
            
            # Reached goal
            if abs(current_pos[0] - self.goal[0]) < 1 and abs(current_pos[1] - self.goal[1]) < 1:
                elapsed_time = time.time() - start_time
                print(f"APF found path!")
                print(f"Path length: {len(path)}")
                print(f"Run time: {elapsed_time:.4f} seconds")
                
                if visualize:
                    # 添加最终路径帧
                    frames.append((explored_nodes, path))
                    # 创建动画
                    if len(frames) > 1:
                        self.create_animation(frames, algorithm_name, "APF_Search_Process")
                    # 保存最终结果图片
                    self.save_single_image(explored_nodes, path, 
                                         "APF Final Path", algorithm_name, "final")
                
                return path, explored_nodes, elapsed_time
        
        elapsed_time = time.time() - start_time
        print("APF did not find a path!")
        
        if visualize and frames:
            self.create_animation(frames, algorithm_name, "APF_Search_Process")
        
        return None, explored_nodes, elapsed_time
    
    def compare_algorithms(self):
        """Compare performance of all algorithms"""
        print("\n" + "="*60)
        print("Algorithm Performance Comparison Analysis")
        print("="*60)
        
        algorithms = {
            'BFS': self.bfs,
            'A*': self.astar,
            'Dijkstra': self.dijkstra,
            'GA': self.genetic_algorithm,
            'APF': self.artificial_potential_field
        }
        
        results = {}
        
        for name, algo_func in algorithms.items():
            print(f"\nRunning {name}...")
            try:
                path, explored, runtime = algo_func(visualize=True)  # 启用可视化
                if path:
                    results[name] = {
                        'path_length': len(path),
                        'explored_nodes': len(explored) if explored else 0,
                        'runtime': runtime,
                        'success': True
                    }
                else:
                    results[name] = {'success': False}
            except Exception as e:
                print(f"{name} running error: {e}")
                results[name] = {'success': False}
        
        return results


def create_complex_map():
    """Create complex and difficult map"""
    # Create 30x30 map
    size = 30
    grid = np.zeros((size, size))
    
    # Add complex obstacle patterns
    
    # 1. Outer walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    
    # 2. Spiral obstacles
    for i in range(5, 25):
        if i < 15:
            grid[i, i] = 1
            grid[i, 30-i] = 1
        else:
            grid[i, 15] = 1
            grid[15, i] = 1
    
    # 3. Random obstacle clusters
    np.random.seed(42)
    for _ in range(100):
        x = np.random.randint(1, size-1)
        y = np.random.randint(1, size-1)
        if (x, y) != (1, 1) and (x, y) != (size-2, size-2):  # Avoid blocking start and goal
            grid[x, y] = 1
    
    # 4. Narrow passages
    for i in range(10, 20):
        grid[10, i] = 1
        grid[20, i] = 1
    for i in range(5, 15):
        grid[i, 15] = 1
    
    return grid


def create_simple_map():
    """Create simple map for basic testing"""
    grid = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    return grid


def plot_comprehensive_comparison(results):
    """绘制全面的性能对比图"""
    # 准备数据
    algorithms = []
    path_lengths = []
    runtimes = []
    explored_counts = []
    
    for name, result in results.items():
        if result.get('success', False):
            algorithms.append(name)
            path_lengths.append(result['path_length'])
            runtimes.append(result['runtime'])
            explored_counts.append(result['explored_nodes'])
    
    if not algorithms:
        print("No successful algorithms to compare!")
        return
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 路径长度对比
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    axes[0, 0].bar(algorithms, path_lengths, color=colors[:len(algorithms)])
    axes[0, 0].set_title('Path Length Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Path Length', fontsize=12)
    axes[0, 0].set_xlabel('Algorithm', fontsize=12)
    for i, v in enumerate(path_lengths):
        axes[0, 0].text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # 2. 运行时间对比（对数坐标）
    axes[0, 1].bar(algorithms, runtimes, color=colors[:len(algorithms)])
    axes[0, 1].set_title('Runtime Comparison (Log Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Time (seconds)', fontsize=12)
    axes[0, 1].set_xlabel('Algorithm', fontsize=12)
    axes[0, 1].set_yscale('log')
    for i, v in enumerate(runtimes):
        axes[0, 1].text(i, v * 1.1, f'{v:.4f}s', ha='center', fontweight='bold')
    
    # 3. 探索节点数对比
    axes[1, 0].bar(algorithms, explored_counts, color=colors[:len(algorithms)])
    axes[1, 0].set_title('Explored Nodes Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Number of Nodes', fontsize=12)
    axes[1, 0].set_xlabel('Algorithm', fontsize=12)
    for i, v in enumerate(explored_counts):
        axes[1, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # 4. 综合性能评分（归一化后的加权和，越小越好）
    max_length = max(path_lengths)
    max_time = max(runtimes)
    max_explored = max(explored_counts)
    
    scores = []
    for i, alg in enumerate(algorithms):
        norm_length = path_lengths[i] / max_length
        norm_time = runtimes[i] / max_time
        norm_explored = explored_counts[i] / max_explored
        score = norm_length + norm_time + norm_explored
        scores.append(score)
    
    axes[1, 1].bar(algorithms, scores, color=colors[:len(algorithms)])
    axes[1, 1].set_title('Comprehensive Score (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_xlabel('Algorithm', fontsize=12)
    for i, v in enumerate(scores):
        axes[1, 1].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存对比图
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/performance_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPerformance comparison chart saved as 'results/performance_comparison.png'")
    plt.close()


def main():
    """Main function"""
    print("="*60)
    print("Robot Obstacle Avoidance Path Planning System")
    print("="*60)
    
    # 创建结果文件夹
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Select map
    print("\nSelect map type:")
    print("1. Simple Map")
    print("2. Complex High-Difficulty Map")
    
    choice = input("Please enter choice (1/2, default 1): ").strip()
    
    if choice == '2':
        grid = create_complex_map()
        start = (1, 1)
        goal = (28, 28)
        print("Using complex high-difficulty map")
    else:
        grid = create_simple_map()
        start = (0, 0)
        goal = (9, 9)
        print("Using simple map")
    
    print(f"Map size: {grid.shape}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    
    # Create planner
    planner = RobotPathPlanning(grid, start, goal)
    
    # 保存初始地图
    initial_grid_display = grid.copy().astype(float)
    initial_grid_display[start[0], start[1]] = 4
    initial_grid_display[goal[0], goal[1]] = 5
    plt.figure(figsize=(10, 8))
    plt.imshow(initial_grid_display, cmap=planner.cmap, interpolation='nearest')
    plt.title('Initial Map')
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5],
                label=['Land', 'Obstacle', 'Explored', 'Path', 'Start', 'Goal'])
    plt.savefig('results/initial_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Initial map saved as 'results/initial_map.png'")
    
    # Run various algorithms
    print("\nStarting path planning...")
    print("="*60)
    
    # 运行所有算法并收集结果
    results = planner.compare_algorithms()
    
    # 绘制综合对比图
    plot_comprehensive_comparison(results)
    
    # 打印详细结果表格
    print("\n" + "="*80)
    print("Detailed Results Table")
    print("="*80)
    print(f"{'Algorithm':<15} {'Path Length':<12} {'Explored Nodes':<15} {'Runtime(s)':<15} {'Success':<10}")
    print("-"*80)
    
    for name, result in results.items():
        if result.get('success', False):
            print(f"{name:<15} {result['path_length']:<12} {result['explored_nodes']:<15} "
                 f"{result['runtime']:<15.4f} {'✓':<10}")
        else:
            print(f"{name:<15} {'-':<12} {'-':<15} {'-':<15} {'✗':<10}")
    
    # Analysis summary
    print("\n" + "="*60)
    print("Algorithm Analysis Summary")
    print("="*60)
    print("""
    📊 Key Findings:
    - A* algorithm achieves the best balance between path optimality and search efficiency
    - BFS and Dijkstra guarantee optimal paths but explore more nodes
    - GA shows fast convergence but may not guarantee optimality
    - APF is sensitive to parameter tuning and may fail in complex environments
    
    💡 Recommendations:
    - For static maps: Use A* algorithm (best overall performance)
    - For unweighted graphs: Use BFS (simplest implementation)
    - For weighted graphs: Use Dijkstra (handles different movement costs)
    - For complex optimization: Use GA (good for high-dimensional problems)
    - For dynamic environments: Consider improving APF or using hybrid methods
    
    🎯 Improvement Directions:
    - Hybrid algorithms combining A* with GA
    - Dynamic window approach with local replanning
    - Deep reinforcement learning for complex environments
    - Multi-objective optimization considering both path length and safety
    """)
    
    print("\n" + "="*60)
    print("✅ All results saved to the 'results' folder!")
    print("   - GIF animations show the search process")
    print("   - Final path images show the optimal paths")
    print("   - Performance comparison chart summarizes all metrics")
    print("="*60)


if __name__ == "__main__":
    # Set random seeds for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run main program
    main()