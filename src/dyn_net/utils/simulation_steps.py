import numpy as np

from dyn_net.dynamical_systems.state_transform import get_state_transform
from dyn_net.dynamical_systems.system_bundle import get_system_bundle
from dyn_net.integrator.params import EulerMaruyamaParams
from dyn_net.noise import get_noise
from dyn_net.utils.initial_condition import build_initial_condition
from dyn_net.utils.network import build_network_from_config
from dyn_net.utils.validation import validate_config


def prepare_network(config: dict):
    validate_config(config)
    return build_network_from_config(config)


def prepare_system(config: dict, A):
    system_cfg = config["system"]
    system_params = dict(system_cfg.get("params", {}))
    system_params["A"] = A
    return get_system_bundle(system_cfg["name"], system_params)


def prepare_noise(config: dict):
    noise_cfg = config["noise"]
    _, pG = get_noise(noise_cfg["name"], noise_cfg.get("params", {}))
    return pG


def prepare_integrator(config: dict):
    return EulerMaruyamaParams.model_validate(config["integrator"])


def prepare_rng(config: dict):
    run_seed = config.get("run", {}).get("seed")
    rng = np.random.default_rng(run_seed)
    if run_seed is not None:
        np.random.seed(int(run_seed))
    return rng


def prepare_initial_condition(config: dict, n: int, rng):
    return build_initial_condition(config, n, rng)


def prepare_state_transform(system_name: str):
    return get_state_transform(system_name)
