def find_lowest_cost_node(costs):
    

graph = {}
graph['start'] = {}
graph['start']['a'] = 6
graph['start']['b'] = 2
graph['a'] = {}
graph['finish'] = {}
graph['a']['finish'] = 1
graph['b'] = {}
graph['b']['finish'] = 5
graph['b']['a'] = 3

infinity = float('inf')
costs = {}
costs['a'] = 6
costs['b'] = 2
costs['finish'] = infinity

parents = {}
parents['a'] = 'start'
parents['b'] = 'start'
parents['finish'] = None

processed = set()

node = find_lowest_cost_node(costs)
while node is not None:
    cost = costs[node]
    neighbors = graph[node]
    for n in neighbors.keys():
        new_cost = cost + neighbors[n]  
        if costs[n] > new_cost:
            costs = new_cost
            parents[n] = node
    processed.add(node)
    node = find_lowest_cost_node(costs)
print(graph)