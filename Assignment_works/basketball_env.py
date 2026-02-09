"""
Basketball Shooter Environment for Reinforcement Learning
Compatible with Gymnasium and Stable-Baselines3
"""
import pygame
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 700
FPS = 60
BALL_RADIUS = 8
HOOP_X, HOOP_Y = 800, 150
HOOP_RADIUS = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)
GREEN = (34, 139, 34)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)


class BasketballShooterEnv(gym.Env):
    """Custom Environment for Basketball Shooting Game"""
    metadata = {"render_modes": ["human", "no_render"], "render_fps": FPS}
    
    def __init__(self, render_mode=None, use_perfect_shots=False, perfect_shots=None):
        super().__init__()
        self.render_mode = render_mode
        self.use_perfect_shots = use_perfect_shots
        self.perfect_shots = perfect_shots if perfect_shots else {}
        self.screen = None
        self.clock = None
        self.font = None
        self.gravity = 0.3
        
        # Action space: [angle, power]
        # angle: -1 to 1 maps to 20° to 80°
        # power: -1 to 1 maps to 10 to 30
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Observation space: [norm_x, norm_y, dx, dy]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -2, -2], dtype=np.float32),
            high=np.array([1, 1, 2, 2], dtype=np.float32),
            dtype=np.float32
        )

        self.score = 0
        self.total_shots = 0
        
    def reset_shot(self):
        """Reset ball position for a new shot"""
        self.start_x = random.randint(100, 400)
        self.start_y = HEIGHT - 100
        self.ball_x = self.start_x
        self.ball_y = self.start_y
        self.trajectory = [(self.ball_x, self.ball_y)]
        self.shot_complete = False
        self.scored = False
        self.frame_count = 0
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.reset_shot()
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation state"""
        norm_x = self.ball_x / WIDTH
        norm_y = self.start_y / HEIGHT
        dx = (HOOP_X - self.ball_x) / WIDTH
        dy = (HOOP_Y - self.start_y) / HEIGHT
        return np.array([norm_x, norm_y, dx, dy], dtype=np.float32)
    
    def get_perfect_shot(self):
        """Get perfect shot parameters from calibration data"""
        if not self.perfect_shots:
            return 50.0, 22.0
        
        # Check if exact position exists
        if self.start_x in self.perfect_shots:
            return self.perfect_shots[self.start_x]
        
        # Find two nearest positions and interpolate
        positions = sorted([p for p in self.perfect_shots.keys() if isinstance(p, int)])
        
        if not positions:
            return 50.0, 22.0
        
        if self.start_x < positions[0]:
            return self.perfect_shots[positions[0]]
        if self.start_x > positions[-1]:
            return self.perfect_shots[positions[-1]]
        
        # Linear interpolation
        for i in range(len(positions) - 1):
            if positions[i] <= self.start_x <= positions[i + 1]:
                x1, x2 = positions[i], positions[i + 1]
                angle1, power1 = self.perfect_shots[x1]
                angle2, power2 = self.perfect_shots[x2]
                
                ratio = (self.start_x - x1) / (x2 - x1)
                angle = angle1 + ratio * (angle2 - angle1)
                power = power1 + ratio * (power2 - power1)
                
                return angle, power
        
        return 50.0, 22.0
    
    def step(self, action):
        """Execute one step in the environment"""
        # Use perfect shots if enabled, otherwise use action
        if self.use_perfect_shots:
            self.angle, self.power = self.get_perfect_shot()
        else:
            self.angle = np.interp(action[0], [-1, 1], [20, 80])
            self.power = np.interp(action[1], [-1, 1], [10, 30])

        angle_rad = math.radians(self.angle)
        self.vx = self.power * math.cos(angle_rad)
        self.vy = -self.power * math.sin(angle_rad)

        terminated = False
        truncated = False
        reward = 0
        
        self.total_shots += 1

        # Simulate shot trajectory frame by frame
        while not self.shot_complete:
            self.vy += self.gravity
            self.ball_x += self.vx
            self.ball_y += self.vy
            
            distance = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
            
            # Reward shaping: encourage getting close to hoop
            if distance < 100:
                reward += 0.5
            if distance < 50:
                reward += 1

            self.trajectory.append((int(self.ball_x), int(self.ball_y)))
            self.frame_count += 1
            
            # Check bounds and timeout
            if self.ball_y > HEIGHT or self.ball_x > WIDTH or self.frame_count > 400:
                self.shot_complete = True
                break
            
            # Check if ball scores in hoop
            if self.ball_y >= HOOP_Y - 30 and self.ball_y <= HOOP_Y + 30 and not self.scored:
                distance = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
                if distance < HOOP_RADIUS:
                    self.scored = True
                    self.score += 1
                    reward = 1000  # Big reward for scoring!
                    self.shot_complete = True
                    break
                elif distance < HOOP_RADIUS * 2:
                    reward += 50  # Close shots get bonus
            
            # Penalty for overshooting
            if self.ball_y < HOOP_Y - 50 and self.ball_x > HOOP_X - 50:
                self.shot_complete = True
                reward = -100
                break

            # Render if in human mode
            if self.render_mode == "human":
                self.render()

        # Penalty based on distance if missed
        if not self.scored and self.shot_complete:
            distance = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
            reward = -100 - (distance / 10)

        terminated = self.shot_complete
        return self._get_observation(), reward, terminated, truncated, {}
    
    def render(self):
        """Render the game visually"""
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Basketball Shooter - RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        # Draw background
        self.screen.fill(GREEN)
        
        # Draw ground
        pygame.draw.rect(self.screen, BROWN, (0, HEIGHT - 50, WIDTH, 50))
        
        # Draw backboard
        backboard_width = 80
        backboard_height = 120
        backboard_x = HOOP_X + 10
        backboard_y = HOOP_Y - 60
        pygame.draw.rect(self.screen, DARK_GRAY, 
                        (backboard_x, backboard_y, backboard_width, backboard_height), 0)
        pygame.draw.rect(self.screen, WHITE, 
                        (backboard_x, backboard_y, backboard_width, backboard_height), 3)
        
        # Draw backboard pole
        pygame.draw.rect(self.screen, GRAY, (HOOP_X + 40, backboard_y + backboard_height, 10, 100))
        
        # Draw 3D hoop
        pygame.draw.ellipse(self.screen, (180, 0, 0), 
                           (HOOP_X - HOOP_RADIUS, HOOP_Y - 8, HOOP_RADIUS * 2, 16), 4)
        pygame.draw.ellipse(self.screen, RED, 
                           (HOOP_X - HOOP_RADIUS + 2, HOOP_Y - 6, HOOP_RADIUS * 2 - 4, 12), 2)
        
        # Draw net
        net_points = 12
        net_depth = 35
        for i in range(net_points):
            angle = (i / net_points) * 2 * math.pi
            x1 = HOOP_X + int(HOOP_RADIUS * 0.9 * math.cos(angle))
            y1 = HOOP_Y + int(4 * math.sin(angle))
            x2 = HOOP_X + int((HOOP_RADIUS - 8) * math.cos(angle))
            y2 = HOOP_Y + net_depth + int(3 * math.sin(angle))
            pygame.draw.line(self.screen, WHITE, (x1, y1), (x2, y2), 1)
        
        pygame.draw.ellipse(self.screen, WHITE, 
                           (HOOP_X - HOOP_RADIUS + 8, HOOP_Y + net_depth - 3, 
                            (HOOP_RADIUS - 8) * 2, 6), 1)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.screen, WHITE, False, self.trajectory, 2)
        
        # Draw basketball
        pygame.draw.circle(self.screen, ORANGE, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS)
        pygame.draw.circle(self.screen, BLACK, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS, 2)
        
        # Draw basketball seams
        for angle_offset in [0, 90]:
            angle = math.radians(angle_offset + self.frame_count * 5)
            x1 = int(self.ball_x + BALL_RADIUS * 0.8 * math.cos(angle))
            y1 = int(self.ball_y + BALL_RADIUS * 0.8 * math.sin(angle))
            x2 = int(self.ball_x - BALL_RADIUS * 0.8 * math.cos(angle))
            y2 = int(self.ball_y - BALL_RADIUS * 0.8 * math.sin(angle))
            pygame.draw.line(self.screen, BLACK, (x1, y1), (x2, y2), 1)
        
        # Draw status text
        status = "SCORED! 🎉" if self.scored else "MISSED!" if self.shot_complete else "SHOOTING..."
        status_color = (0, 255, 0) if self.scored else RED if self.shot_complete else WHITE
        
        status_text = self.font.render(status, True, status_color)
        score_text = self.font.render(f"Score: {self.score}/{self.total_shots}", True, WHITE)
        
        self.screen.blit(status_text, (20, 20))
        self.screen.blit(score_text, (20, 60))
        
        pygame.display.flip()
        self.clock.tick(FPS)
        pygame.event.pump()
    
    def close(self):
        """Clean up resources"""
        if self.screen is not None:
            pygame.quit()
