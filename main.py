import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt

from dp_utils import *

OmegaConf.register_new_resolver("eval", custom_eval)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Plot something
    x = [1, 2, 3, 4]
    y = [1, 4, 9, 16]
    plt.plot(x, y)
    plt.ylabel("some numbers")
    plt.show()


if __name__ == "__main__":
    main()
