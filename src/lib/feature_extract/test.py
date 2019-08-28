from utils.config import Config
from generate_features import GenerateFeatures


def main(config):
    generate_features = GenerateFeatures(config)


if __name__ == "__main__":
    config = Config.parse()
    main(config)
