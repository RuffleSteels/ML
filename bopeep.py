import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ----- Pendulum Parameters -----
g = 9.81
L = 1.0
dt = 0.05
max_torque = 2.0

# ----- Environment -----
class Pendulum:
    def __init__(self):
        self.reset()

    def reset(self):
        self.theta = np.pi  # Start hanging down
        self.omega = 0.0
        return self.get_state()

    def get_state(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.omega], dtype=np.float32)

    def step(self, torque):
        torque = np.clip(torque, -max_torque, max_torque)
        self.omega += ( -3 * g / (2 * L) * np.sin(self.theta + np.pi) + 3.0 / (L**2) * torque ) * dt
        self.omega = np.clip(self.omega, -8, 8)
        self.theta += self.omega * dt
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi  # wrap between [-pi, pi]
        reward = - (self.theta**2 + 0.1*self.omega**2 + 0.001*(torque**2)) / 10.0

        done = False
        return self.get_state(), reward, done

# ----- Value Function Approximator -----
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# ----- Fitted Value Iteration -----
def generate_dataset(env, value_net, i, num_samples=10000):
    states = []
    targets = []
    for _ in range(num_samples):
        env.reset()
        for _ in range(1):
            s = env.get_state()
            best_return = -np.inf
            for torque in np.linspace(-max_torque, max_torque, 11):
                env2 = Pendulum()
                env2.theta, env2.omega = env.theta, env.omega
                s_next, r, _ = env2.step(torque)
                s_next_tensor = torch.tensor(s_next, dtype=torch.float32)
                with torch.no_grad():
                    future_v = value_net(s_next_tensor).item()
                total_return = r if i == 0 else r + gamma * future_v
                best_return = max(best_return, total_return)
            states.append(s)
            targets.append(best_return)

    targets = np.array(targets, dtype=np.float32)
    targets = np.clip(targets, -50.0, 50.0)  # safety clamp
    return torch.tensor(np.array(states), dtype=torch.float32), torch.tensor(targets).unsqueeze(1)

# ----- Policy -----
def get_action(state):
    best_torque = 0
    best_value = -np.inf
    for torque in np.linspace(-max_torque, max_torque, 11):
        env2 = Pendulum()
        env2.theta, env2.omega = state
        s_next, r, _ = env2.step(torque)
        s_next_tensor = torch.tensor(s_next, dtype=torch.float32)
        with torch.no_grad():
            value = value_net(s_next_tensor).item()
        if value > best_value:
            best_value = value
            best_torque = torque
    return best_torque

# ----- Training -----
gamma = 0.99
value_net = ValueNetwork()
optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
env = Pendulum()

print("Training value function...")
for i in range(20):
    states, targets = generate_dataset(env, value_net, i, num_samples=2000)
    dataset = torch.utils.data.TensorDataset(states, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(3):
        for xb, yb in loader:
            v_pred = value_net(xb)
            loss = nn.MSELoss()(v_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f"Iter {i}, Loss: {loss.item():.4f}")

# ----- Pygame Visualization -----
pygame.init()
size = width, height = 500, 500
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
running = True
env.reset()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Select action
    s = env.get_state()
    torque = get_action((env.theta, env.omega))
    env.step(torque)

    # Clear screen
    screen.fill((255, 255, 255))

    # Draw pendulum
    origin = np.array([width//2, height//2])
    end = origin + np.array([L * np.sin(env.theta), L * np.cos(env.theta)]) * 200
    pygame.draw.line(screen, (0, 0, 0), origin, end.astype(int), 5)
    pygame.draw.circle(screen, (0, 0, 255), end.astype(int), 10)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
