import _init_paths
from lib.feature_extract.generate_features import GenerateFeatures
from lib.feature_extract.create_RDMs import CreateRDMs
from lib.evaluation.evaluate_results import Evaluate


from lib.utils.config import Config


def main(config):
    # GenerateFeatures(config).run()
    # CreateRDMs(config).run()
    Evaluate(config).run()


if __name__ == "__main__":
    config = Config().parse()
    print(config)
    main(config)
