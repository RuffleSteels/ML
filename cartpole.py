from __future__ import division, print_function
from env import CartPole, Physics
import numpy as np

def is_terminal(state):
    x, x_dot, theta, theta_dot = state
    return not (
        -3 <= x <= 3 and
        -3 <= x_dot <= 3 and
        -45 * np.pi / 180 <= theta <= 45 * np.pi / 180 and
        -3 <= theta_dot <= 3
    )

def phi(s):
    # x, x_dot, theta, theta_dot = s
    # return np.array([
    #     x, x_dot, theta, theta_dot, 1.0,
    #     x**2, theta**2, theta_dot**2, x_dot**2
    # ])
    x, x_dot, theta, theta_dot = s
    return np.array([
        x / 2.4,                 # normalize to [-1, 1]
        x_dot / 2.0,             # approximate range
        theta / (12 * np.pi / 180),  # to [-1, 1]
        theta_dot / 2.0,
        1.0,
        (x / 2.4)**2,
        (theta / (12 * np.pi / 180))**2,
        np.sin(theta),
        np.cos(theta)
    ])

def value_function(state, theta):
    return phi(state) @ theta

def fit_model(cart_pole, n_states, time_steps):
    states = cart_pole.sample_states(n_states)
    X_list = []
    Y_list = []

    for s in states:
        for t in range(time_steps):
            if is_terminal(s):
                break
            a = np.random.randint(0, 2)
            s_prime = cart_pole.simulate(a, s)
            features = np.concatenate((np.array(s), [a]))  # input: state + action
            X_list.append(features)
            Y_list.append(s_prime)  # output: next state
            s = s_prime

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)

    bias = np.ones((X.shape[0], 1))
    X_aug = np.hstack([X, bias])

    # Initialize theta for linear model: output_dim x input_dim (4 x 6)
    model_theta = np.random.uniform(-0.01, 0.01, size=(4, X_aug.shape[1]))

    # Gradient descent to fit model
    for i in range(100000):
        Y_pred = X_aug @ model_theta.T
        grad = (2 / X_aug.shape[0]) * (Y_pred - Y).T @ X_aug
        old_theta = model_theta.copy()
        model_theta -= 0.01 * grad

        if np.linalg.norm(model_theta - old_theta) < 1e-6:
            print("Model converged after", i, "iterations.")
            break

    return model_theta

def simulate_model(state, action, model_theta, noise_scale=0.01, n_samples=20):
    features = np.concatenate((np.array(state), [action]))
    features = np.append(features, 1.0)  # add bias
    mean = features @ model_theta.T
    samples = mean + noise_scale * np.random.randn(n_samples, 4)

    # Filter out terminal states
    valid_samples = [s for s in samples if not is_terminal(s)]
    return valid_samples if valid_samples else [state]  # fallback to current state if all were bad



def policy(state, model_theta, value_theta, n_samples=500):
    q_values = []
    for a in [0, 1]:
        next_states = simulate_model(state, a, model_theta, n_samples=n_samples)
        expected_v = np.mean([value_function(s_prime, value_theta) for s_prime in next_states])
        q = reward(state) + 0.95 * expected_v
        q_values.append(q)
    return np.argmax(q_values)
def closeness_score(theta, max_theta):
    theta = np.clip(theta, 0, max_theta)
    score = np.cos((theta / max_theta) * (np.pi / 2))
    return score
def reward(state):
    x, x_dot, theta, theta_dot = state
    return (closeness_score(abs(theta), 12 * (np.pi / 180))*5 +
            closeness_score(abs(x), 2)*8 +
            closeness_score(abs(x_dot), 2)*3 +
            closeness_score(abs(theta_dot), 2)*3)

def minibatch_gradient_descent(Phi, Y, theta, batch_size=32, lr=0.1, max_steps=10000):
    n_samples = Phi.shape[0]
    theta_old = theta.copy()

    for step in range(max_steps):
        # Shuffle indices
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = indices[start_idx:end_idx]

            Phi_batch = Phi[batch_idx]
            Y_batch = Y[batch_idx]

            predictions = Phi_batch @ theta
            errors = predictions - Y_batch
            grad = (2 / len(batch_idx)) * Phi_batch.T @ errors

            theta -= lr * grad

        # Check for convergence at the end of full epoch
        if np.linalg.norm(theta - theta_old) < 1e-6:
            print(f"[✓] Mini-batch GD converged after {step} epochs.")
            break

        theta_old = theta.copy()

    return theta

def fitted_value_iteration(cart_pole, model_theta, states, gamma=0.95, k_samples=5000, n_iterations=600):
    """
    Implements fitted value iteration for continuous state MDPs using linear function approximation.
    """
    n_features = len(phi(states[0]))  # Determine feature dimensionality
    theta = np.zeros(n_features)      # Initialize theta = 0

    for it in range(n_iterations):
        Y = np.zeros(len(states))         # Targets y^(i)
        Phi = np.zeros((len(states), n_features))  # Feature matrix

        for i, s in enumerate(states):
            # if is_terminal(s):
            #     continue
            q_values = []

            for a in [0, 1]:
                next_states = simulate_model(s, a, model_theta, n_samples=k_samples)
                estimated_values = [reward(s) + gamma * value_function(s_prime, theta) for s_prime in next_states]
                q_a = np.mean(estimated_values)
                q_values.append(q_a)

            Y[i] = max(q_values)
            Phi[i] = phi(s)

        theta_old = theta.copy()
        theta = minibatch_gradient_descent(Phi, Y, theta, batch_size=1000, lr=0.0001, max_steps=10000)

        print(np.linalg.norm(theta_old - theta))
        if np.linalg.norm(theta_old - theta) < 1e-6:
            print(f"[✓] Converged after {it} iterations.")
            break


    return theta

def main():
    # Simulation parameters

    cart_pole = CartPole(Physics())

    consecutive_no_learning_trials = 0
    n_trials = 20
    t_time_steps = 100
    # model_theta = fiit_model(cart_pole, n_trials, t_time_steps)

    k_samples = 20
    states = cart_pole.sample_states(20)

    gamma = 0.95

    # theta = np.zeros((len(states),len(states)))
    #
    # Y = np.zeros((len(states),4))
    # for i in range(len(states)):
    #     q_a = np.zeros((2,4))
    #
    #     for action in [0, 1]:
    #         new_state = cart_pole.simulate(action, states[i])
    #         mean = value_function(states[i], action, model_theta)
    #         diff = np.array(new_state) - mean
    #         diff = diff[:, None]
    #         model_covariance = (1/n_trials)*(1/(t_time_steps-1))*(diff @ diff.T)
    #
    #         samples = np.random.multivariate_normal(mean, model_covariance, size=k_samples)
    #
    #         q_a[action] = (1/k_samples) * reward(states[i]) * sum(gamma * value_function(state_prime, action, model_theta) for state_prime in samples)
    #     Y[i] = np.max(q_a, axis=0)
    #
    # S = np.array(states)
    # S_l = len(states)
    # for _ in range(100000000):
    #     old_theta = theta.copy()
    #     Y_pred = theta @ S
    #     grad = (1 / S_l) * (Y - Y_pred) @ S.T
    #     theta += 0.001 * grad
    #
    #     if np.linalg.norm(theta - old_theta) < 1e-6:
    #         print("Theta converged after", _, "iterations.")
    #         break
    #
    # print(theta)
    x = np.random.uniform(-1.5, 1.5)
    x_dot = np.random.uniform(-1, 1)
    theta = np.random.uniform(-0.1, 0.1)
    theta_dot = np.random.uniform(-1, 1)
    state_tuple = (x, x_dot, theta, theta_dot)

    states = cart_pole.sample_states(50)
    model_theta = fit_model(cart_pole, n_states=500, time_steps=200)
    value_theta = fitted_value_iteration(cart_pole, model_theta, states, gamma=0.95, k_samples=50)

    print(simulate_model(state_tuple, 0, model_theta, n_samples=2))
    print(cart_pole.simulate(0, state_tuple))

    # value_theta = fitted_value_iteration(cart_pole, model_theta, n_samples=50, n_iterations=1000)

    from renderer import PygameRenderer  # or wherever you placed it

    print("[INFO] Training complete. Running final policy in a pygame window...")

    renderer = PygameRenderer()

    x = np.random.uniform(-1.5, 1.5)
    x_dot = np.random.uniform(-1, 1)
    theta = np.random.uniform(-0.1, 0.1)
    theta_dot = np.random.uniform(-1, 1)
    state_tuple = (x, x_dot, theta, theta_dot)

    while True:
        x, x_dot, theta, theta_dot = state_tuple
        renderer.draw(state_tuple)
        a = policy(state_tuple, model_theta, value_theta)
        state_tuple = cart_pole.simulate(a, state_tuple)

        if (abs(x) > 3):
            x = np.random.uniform(-1.5, 1.5)
            x_dot = np.random.uniform(-1, 1)
            theta = np.random.uniform(-0.1, 0.1)
            theta_dot = np.random.uniform(-1, 1)

            state_tuple = (x, x_dot, theta, theta_dot)

if __name__ == '__main__':
    main()
