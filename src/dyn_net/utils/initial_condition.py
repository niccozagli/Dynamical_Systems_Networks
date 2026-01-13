from dyn_net.dynamical_systems.initial_conditions import get_initial_condition_builder


def build_initial_condition(config: dict, n, rng):
    system_name = config["system"]["name"]
    ic_cfg = config.get("initial_condition", {})
    builder = get_initial_condition_builder(system_name)
    return builder(ic_cfg, int(n), rng)
