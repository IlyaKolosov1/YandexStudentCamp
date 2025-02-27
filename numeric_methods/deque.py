from collections import deque


graph = {}
graph['you'] = ['seller_lulu', 'wiki', 'robb']
graph['seller_lulu'] = ['pete', 'seller_cpp']
graph['wiki'] = []
graph['robb'] = []
graph['pete'] =[]
graph['seller_cpp'] = []

def person_is_seller(person):
    return person[:6] == 'seller'

def search(graph, name):
    search_deq = deque()
    search_deq += graph[name]
    searched = []
    while search_deq:
        person = search_deq.popleft()
        if not person in searched:
            if person_is_seller(person):
                print(f'{person} is seller')
                search_deq += graph[person]
                
            else:
                search_deq += graph[person]
                searched.append(person)
    return False

search(graph=graph, name='you')
