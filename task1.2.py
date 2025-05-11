import networkx as nx
import random
import matplotlib.pyplot as plt

# 1. Сформировать список пользователей

def generate_users(num_users=50):

    users = []
    for i in range(num_users):
        user_id = f"u{i+1:02}"
        age = random.randint(16, 65)
        gender = random.choice(["male", "female"])
        education_level = random.choice(["high_school", "bachelor", "master"])
        messages_per_day = random.randint(0, 100)
        avg_message_length = random.randint(5, 200)
        positive_ratio = random.uniform(0.0, 1.0)
        is_active = random.choice([0, 1])
        friends = []  # Пока пустой список, заполним позже
        users.append({
            "user_id": user_id,
            "age": age,
            "gender": gender,
            "education_level": education_level,
            "messages_per_day": messages_per_day,
            "avg_message_length": avg_message_length,
            "positive_ratio": positive_ratio,
            "is_active": is_active,
            "friends": friends
        })
    return users


def connect_users(users, avg_friends=5):

    num_users = len(users)
    for user in users:
        # Определяем, сколько друзей будет у пользователя.  
        num_friends = min(random.randint(0, avg_friends * 2), num_users - 1)  # Ограничиваем число друзей

        # Выбираем случайных друзей, исключая самого себя и тех, кто уже в списке друзей.
        possible_friends = [u["user_id"] for u in users if u["user_id"] != user["user_id"] and u["user_id"] not in user["friends"]]
        
        # Выбираем случайных друзей из возможных
        friends = random.sample(possible_friends, min(num_friends, len(possible_friends)))
        user["friends"] = friends

        # Обеспечиваем взаимность дружбы (если А дружит с B, то B дружит с A)
        for friend_id in friends:
            friend = next((u for u in users if u["user_id"] == friend_id), None)
            if friend and user["user_id"] not in friend["friends"]:
                friend["friends"].append(user["user_id"])

    return users


# Генерируем список пользователей
users = generate_users(num_users=50)

# Устанавливаем дружеские связи
users = connect_users(users, avg_friends=5)


# 2. Построить социальный граф

def build_social_graph(users):
    graph = nx.Graph()
    # Добавляем узлы (пользователей) в граф
    for user in users:
        graph.add_node(user["user_id"], data=user)
    # Добавляем ребра (связи между друзьями)
    for user in users:
        for friend_id in user["friends"]:
            if graph.has_node(friend_id):  # Убедимся, что друг существует в графе.
                graph.add_edge(user["user_id"], friend_id)

    return graph
graph = build_social_graph(users)

# Визуализация графа с цветами для активных/неактивных пользователей
def visualize_graph(graph):
    node_colors = ['green' if graph.nodes[node]['data']['is_active'] == 1 else 'red' for node in graph.nodes()]
    plt.figure(figsize=(12, 8))  # Увеличиваем размер графика для лучшей читаемости
    # Используем spring_layout для более читаемого расположения узлов
    pos = nx.spring_layout(graph, seed=42)  # seed для воспроизводимости
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, font_color="black")
    plt.title("Социальный граф (зеленый - активные, красный - неактивные)")
    plt.show()
visualize_graph(graph)

# 3. Выполнить базовый графовый анализ
def analyze_graph(graph):
    # 3.1 Пользователи с наибольшим числом связей (по степени)
    degree_centrality = nx.degree_centrality(graph)
    top_users_degree = sorted(degree_centrality.items(),
key=lambda x: x[1], reverse=True)[:5]
    print("\nТоп 5 пользователей по степени (числу связей):")
    for user_id, centrality in top_users_degree:
        print(f"  {user_id}: {centrality:.3f}")
    # 3.2 "Центральные" пользователи (по степени, посредничеству или кластерному коэффициенту)
    # Центральность по посредничеству (Betweenness Centrality)
    betweenness_centrality = nx.betweenness_centrality(graph)
    top_users_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nТоп 5 пользователей по посредничеству:")
    for user_id, centrality in top_users_betweenness:
        print(f"  {user_id}: {centrality:.3f}")
    # Кластерный коэффициент
    clustering_coefficient = nx.clustering(graph)
    top_users_clustering = sorted(clustering_coefficient.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nТоп 5 пользователей по кластерному коэффициенту:")
    for user_id, coefficient in top_users_clustering:
        print(f"  {user_id}: {coefficient:.3f}")

    # 3.3 Определить, есть ли изолированные кластеры (группы)
    connected_components = list(nx.connected_components(graph))
    num_clusters = len(connected_components)
    print(f"\nКоличество изолированных кластеров: {num_clusters}")
    if num_clusters > 1:
        print("Размеры кластеров:", [len(cluster) for cluster in connected_components])
analyze_graph(graph)
