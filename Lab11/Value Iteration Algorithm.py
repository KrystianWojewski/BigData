import numpy as np
import gym

def value_iteration(env, gamma=0.99, epsilon=1e-6):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Inicjalizacja wartości stanów
    V = np.zeros(num_states)

    while True:
        delta = 0

        for s in range(num_states):
            v = V[s]
            q_values = []

            # Obliczanie wartości dla każdej akcji w danym stanie
            for a in range(num_actions):
                q_value = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    q_value += prob * (reward + gamma * V[next_state])
                q_values.append(q_value)

            # Aktualizacja wartości stanu na podstawie maksymalnej wartości akcji
            V[s] = max(q_values)

            delta = max(delta, abs(v - V[s]))

        if delta < epsilon:
            break

    # Znalezienie optymalnej polityki na podstawie estymowanych wartości stanów
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        q_values = []
        for a in range(num_actions):
            q_value = 0
            for prob, next_state, reward, done in env.P[s][a]:
                q_value += prob * (reward + gamma * V[next_state])
            q_values.append(q_value)
        policy[s] = np.argmax(q_values)

    return V, policy


# Tworzenie środowiska Frozen Lake
env = gym.make("FrozenLake8x8-v1", render_mode="human")

# Wywołanie algorytmu Value Iteration
values, optimal_policy = value_iteration(env)

# Wyświetlenie estymowanych wartości stanów
print("Estymowane wartości stanów:")
print(values)

# Wyświetlenie optymalnej polityki
print("Optymalna polityka:")
print(optimal_policy)

env.reset()

done = False

while not done:
    for policy in optimal_policy:
        _, _, done, _, _ = env.step(policy)
