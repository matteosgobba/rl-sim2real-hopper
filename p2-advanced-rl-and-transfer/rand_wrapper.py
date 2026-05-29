import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper for Uniform Domain Randomization (UDR) and Adaptive Domain Randomization (ADR) on the PandaPush object mass.

    The mass is sampled and applied after env.reset(), because reset may restore default object dynamics.
    """

    def __init__(
        self,
        env,
        mode: str = "none",
        mass_range: Tuple[float, float] = (1.0, 5.0),
        initial_mass_range: Tuple[float, float] = (1.0, 1.5),
        mass_limits: Tuple[float, float] = (0.5, 6.0),
        adr_step: float = 0.25,
        boundary_prob: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(env)

        if mode not in ["none", "udr", "adr"]:
            raise ValueError(f"Unsupported randomization mode: {mode}")

        self.mode = mode
        self.verbose = verbose

        self.mass_range = mass_range

        self.mass_min, self.mass_max = initial_mass_range
        self.mass_min_limit, self.mass_max_limit = mass_limits
        self.adr_step = adr_step
        self.boundary_prob = boundary_prob

        self.current_mass: Optional[float] = None
        self.last_sample_type: Optional[str] = None
        self.last_success: Optional[bool] = None
        self.episode_count = 0

    def _sample_mass(self) -> Optional[float]:
        if self.mode == "none":
            self.last_sample_type = "none"
            return None

        if self.mode == "udr":
            self.last_sample_type = "uniform"
            return float(np.random.uniform(self.mass_range[0], self.mass_range[1]))

        if self.mode == "adr":
            return self._sample_mass_adr()

        raise ValueError(f"Unsupported mode: {self.mode}")

    def _sample_mass_adr(self) -> float:
        if random.random() < self.boundary_prob:
            if random.random() < 0.5:
                self.last_sample_type = "lower_boundary"
                return float(self.mass_min)
            else:
                self.last_sample_type = "upper_boundary"
                return float(self.mass_max)

        self.last_sample_type = "uniform"
        return float(np.random.uniform(self.mass_min, self.mass_max))

    def _update_adr_range(self) -> None:
        if self.mode != "adr":
            return

        if self.last_success is None or self.last_sample_type is None:
            return

        if self.last_sample_type == "upper_boundary":
            if self.last_success:
                self.mass_max += self.adr_step
            else:
                self.mass_max -= 0.5 * self.adr_step

        elif self.last_sample_type == "lower_boundary":
            if self.last_success:
                self.mass_min -= self.adr_step
            else:
                self.mass_min += 0.5 * self.adr_step

        self.mass_min = float(np.clip(self.mass_min, self.mass_min_limit, self.mass_max_limit))
        self.mass_max = float(np.clip(self.mass_max, self.mass_min_limit, self.mass_max_limit))

        if self.mass_min > self.mass_max:
            midpoint = 0.5 * (self.mass_min + self.mass_max)
            self.mass_min = midpoint
            self.mass_max = midpoint

        if abs(self.mass_max - self.mass_min) < 1e-6:
            self.mass_max = min(self.mass_min + 1e-3, self.mass_max_limit)

    def _get_object_body_id(self):
        sim = self.env.unwrapped.task.sim
        return sim, sim._bodies_idx["object"]

    def _set_object_mass(self, mass: float) -> None:
        sim, object_body_id = self._get_object_body_id()

        sim.physics_client.changeDynamics(
            bodyUniqueId=object_body_id,
            linkIndex=-1,
            mass=float(mass),
        )

    def _get_object_mass(self) -> Optional[float]:
        try:
            sim, object_body_id = self._get_object_body_id()
            dynamics_info = sim.physics_client.getDynamicsInfo(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
            )
            return float(dynamics_info[0])
        except Exception:
            return None

    def reset(self, **kwargs):
        self.episode_count += 1

        self._update_adr_range()

        obs, info = self.env.reset(**kwargs)

        new_mass = self._sample_mass()

        if new_mass is not None:
            self._set_object_mass(new_mass)
            actual_mass = self._get_object_mass()

            self.current_mass = float(new_mass)

            info = dict(info)
            info["sampled_mass"] = float(new_mass)
            info["actual_mass"] = actual_mass
            info["randomization_mode"] = self.mode

            if self.mode == "adr":
                info["adr_mass_min"] = float(self.mass_min)
                info["adr_mass_max"] = float(self.mass_max)

            if self.verbose:
                if self.mode == "udr":
                    print(
                        f"[UDR] episode={self.episode_count} "
                        f"sampled_mass={new_mass:.3f} "
                        f"actual_mass={actual_mass} "
                        f"range=[{self.mass_range[0]:.3f}, {self.mass_range[1]:.3f}]"
                    )
                elif self.mode == "adr":
                    print(
                        f"[ADR] episode={self.episode_count} "
                        f"sampled_mass={new_mass:.3f} "
                        f"actual_mass={actual_mass} "
                        f"range=[{self.mass_min:.3f}, {self.mass_max:.3f}] "
                        f"sample={self.last_sample_type}"
                    )

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if done:
            if isinstance(info, dict) and "is_success" in info:
                self.last_success = bool(info["is_success"])
            else:
                self.last_success = False

            if isinstance(info, dict):
                info = dict(info)
                info["sampled_mass"] = self.current_mass
                info["randomization_mode"] = self.mode

                if self.mode == "adr":
                    info["adr_mass_min"] = float(self.mass_min)
                    info["adr_mass_max"] = float(self.mass_max)

        return obs, reward, terminated, truncated, info