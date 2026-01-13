from dyn_net.dynamical_systems.jit_registry import get_jit_kernel
from dyn_net.dynamical_systems.registry import get_drift_params_model
from dyn_net.dynamical_systems.stats_registry import get_stats
from dyn_net.integrator.params import EulerMaruyamaParams
from dyn_net.networks import get_network
from dyn_net.noise import get_noise


def validate_config(config: dict) -> None:
    required = ("system", "network", "noise", "integrator")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required top-level keys: {missing}")

    system_cfg = config["system"]
    network_cfg = config["network"]
    noise_cfg = config["noise"]
    integrator_cfg = config["integrator"]

    get_network(network_cfg["name"], network_cfg.get("params", {}))

    system_params = dict(system_cfg.get("params", {}))
    Params = get_drift_params_model(system_cfg["name"])
    extra = set(system_params) - set(Params.model_fields)
    if extra:
        raise ValueError(f"Unknown system parameters: {sorted(extra)}")
    get_stats(system_cfg["name"])
    get_jit_kernel(system_cfg["name"])
    get_noise(noise_cfg["name"], noise_cfg.get("params", {}))
    EulerMaruyamaParams.model_validate(integrator_cfg)
