from dyn_net.networks import get_network


def build_network_from_config(config: dict):
    network_cfg = config["network"]
    build_net, p_net = get_network(network_cfg["name"], network_cfg.get("params", {}))
    return build_net(p_net)
