from collections import defaultdict
import sys

def parse_dependencies(lines):
    # Создаем граф зависимостей
    graph = defaultdict(list)
    # Множество всех пакетов
    packages = set()
    
    for line in lines:
        if "->" in line:
            # Разбираем строку зависимости
            parts = line.strip().split("->")
            pkg1 = parts[0].strip().strip('"')
            pkg2 = parts[1].split("[")[0].strip().strip('"')
            
            graph[pkg1].append(pkg2)
            packages.add(pkg1)
            packages.add(pkg2)
    
    return graph, packages

def topological_sort(graph, packages):
    # Создаем словарь для отслеживания посещенных вершин
    visited = set()
    # Список для хранения результата
    result = []
    
    def dfs(package):
        if package in visited:
            return
        visited.add(package)
        
        # Сначала обрабатываем зависимости
        for dep in graph[package]:
            dfs(dep)
        
        result.append(package)
    
    # Запускаем DFS для каждого пакета
    for package in packages:
        if package not in visited:
            dfs(package)
    
    # Возвращаем список в обратном порядке
    return result[::-1]

def main():
    # Читаем входные данные
    lines = sys.stdin.readlines()
    
    # Получаем граф зависимостей и список пакетов
    graph, packages = parse_dependencies(lines)
    
    # Получаем отсортированный список пакетов
    sorted_packages = topological_sort(graph, packages)
    
    # Выводим результат
    for package in sorted_packages:
        print(package)

if __name__ == "__main__":
    main()