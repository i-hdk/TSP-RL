from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        num_nodes: int = 20,
        batch_size: int = 128,
        num_draw: int = 6,
        seed: int = 69,
    ):
        """
        Args:
            num_nodes (int, optional): Number of nodes in each generated graph. Defaults to 32.
            batch_size (int, optional): Number of graphs to generate. Defaults to 128.
            num_draw (int, optional): When calling the render num_draw graphs will be rendered. 
                Defaults to 6.
            seed (int, optional): Seed of the environment. Defaults to 69.
            video_save_path (str, optional): When set a video of the interactions with the 
                environment is saved at the set location. Defaults to None.
        """
        
        assert (
            num_draw <= batch_size
        ), "Num_draw needs to be equal or lower than the number of generated graphs."

        np.random.seed(seed)

        self.step_count = 0
        self.num_nodes = num_nodes
        self.batch_size = batch_size

        # init video recorder
        self.draw_idxs = np.random.choice(batch_size, num_draw, replace=False)
        self.video_save_path = None

        self.generate_graphs()

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, actions: np.ndarray) -> Tuple[ObsType, float, bool, dict]:
        """
        Run the environment one timestep. It's the users responsiblity to
        call reset() when the end of the episode has been reached. Accepts
        an actions and return a tuple of (observation, reward, done, info)

        Args:
            actions (nd.ndarray): Which node to visit for each graph.
                Shape of actions is (batch_size, 1).

        Returns:
            Tuple[ObsType, float, bool, dict]: Tuple of the observation,
                reward, done and info. The observation is within
                self.observation_space. The reward is for the previous action.
                If done equals True then the episode is over. Stepping through
                environment while done returns undefined results. Info contains
                may contain additions info in terms of metrics, state variables
                and such.
        """
        assert (
            actions.shape[0] == self.batch_size
        ), "Number of actions need to equal the number of generated graphs."

        self.step_count += 1

        # visit each next node
        self.visited[np.arange(len(actions)), actions.T] = 1
        traversed_edges = np.hstack([self.current_location, actions]).astype(int)
        self.sampler.visit_edges(traversed_edges)

        self.current_location = np.array(actions)

        if self.video_save_path is not None:
            self.vid.capture_frame()

        done = self.is_done()
        return (
            self.get_state(),
            -self.sampler.get_distances(traversed_edges),
            done,
            None,
        )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
