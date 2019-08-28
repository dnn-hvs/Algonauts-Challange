import _init_paths
from lib.feature_extract.generate_features import GenerateFeatures
from lib.feature_extract.create_RDMs import CreateRDMs
from lib.evaluation.evaluate_results import Evaluate


from lib.utils.config import Config


def main(config):
    if config.fullblown:
        GenerateFeatures(config).run()
        CreateRDMs(config).run()
        Evaluate(config).run()
        return

    if config.generate_features:
        GenerateFeatures(config).run()
        return

    if config.create_rdms:
        CreateRDMs(config).run()
        return

    if config.evaluate_results:
        print(config.evaluate_results)
        Evaluate(config).run()
        return


if __name__ == "__main__":
    config = Config().parse()
    print(config)
    main(config)
