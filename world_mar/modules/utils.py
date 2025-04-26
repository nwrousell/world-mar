import importlib

def instantiate_from_config(cfg):
    assert "target" in cfg, "this is horribly wrong"

    module, cls = cfg["target"].rsplit('.')
    obj = getattr(importlib.import_module(module), cls)
    return obj(**cfg.get("params", {}))