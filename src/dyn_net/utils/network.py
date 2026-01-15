from dyn_net.networks import get_network
from dyn_net.networks.power_law import PowerLawParams


def build_network_from_config(config: dict):
    network_cfg = config["network"]
    build_net, p_net = get_network(network_cfg["name"], network_cfg.get("params", {}))
    return build_net(p_net)


def update_system_params_for_network(
    system_params: dict,
    network_name: str,
    network_params: dict,
    A=None,
):
    if network_name == "power_law" and "scale" not in system_params:
        p = PowerLawParams.model_validate(network_params)
        rho_n = p.n ** (-p.beta)
        system_params["scale"] = float(rho_n)
    if A is not None:
        system_params["A"] = A
    return system_params
