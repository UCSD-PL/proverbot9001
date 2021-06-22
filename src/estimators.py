import argparse
import torch

from models.features_q_estimator import FeaturesQEstimator


def main():
    parser = \
      argparse.ArgumentParser(
          description="A module using and testing the features q estimator")

    parser.add_argument("estimator_weights")
    parser.add_argument("feature_vals", nargs="+", type=float)
    args = parser.parse_args()
    q_estimator_name, *saved = torch.load(str(args.estimator_weights))
    if q_estimator_name == "features evaluator":
        assert q_estimator_name == "features evaluator"
        q_estimator = FeaturesQEstimator(0, 0, 0)
        q_estimator.load_saved_state(*saved)

        assert len(args.feature_vals) == 7
        word_feature_vals = [int(val) for val in args.feature_vals[:4]]
        vec_feature_vals = args.feature_vals[4:]
        output = q_estimator.model(torch.LongTensor([word_feature_vals]),
                                   torch.FloatTensor([vec_feature_vals]))
    else:
        assert False, "Unsupported estimator type"

    print(output[0].item())


if __name__ == "__main__":
    main()
