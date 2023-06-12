import numpy as np
import gym

def q_learning(env, learning_rate=0.8, discount_factor=0.95, epsilon=0.5, num_epochs=100000):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    Q = np.zeros((num_states, num_actions))

    for epoch in range(num_epochs):
        state, info = env.reset()
        done = False

        while not done:
            # Wybieramy akcję na podstawie wartości Q i strategii epsilon-greedy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Eksploracja
            else:
                action = np.argmax(Q[state, :])  # Eksploatacja

            # Wykonujemy wybraną akcję i obserwujemy nagrodę oraz nowy stan
            next_state, reward, done, _ , info = env.step(action)

            # Aktualizujemy wartość Q dla poprzedniego stanu i akcji
            Q[state, action] = Q[state, action] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

    # Generujemy optymalną politykę na podstawie estymowanych wartości Q
    optimal_policy = np.argmax(Q, axis=1)

    return Q, optimal_policy


# Tworzenie środowiska Frozen Lake
env = gym.make("FrozenLake8x8-v1", render_mode="ansi", is_slippery=False)

# Wywołanie algorytmu Q-learning
Q_values, optimal_policy = q_learning(env)

# Wyświetlenie estymowanych wartości Q
print("Estymowane wartości Q:")
print(Q_values)

# Wyświetlenie optymalnej polityki
print("Optymalna polityka:")
print(optimal_policy)

env = gym.make("FrozenLake8x8-v1", render_mode="human", is_slippery=False)

env.reset()

done = False

while not done:
    for policy in optimal_policy:
        _, _, done, _, _ = env.step(policy)
