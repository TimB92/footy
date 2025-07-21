from argparse import ArgumentParser
from adapters.local_source import LocalDataSource
from domain.preprocess import preprocess
from domain.train import train
from domain.evaluate import predict


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    return parser


def main():
    parser = create_parser()
    args, _ = parser.parse_known_args()
    data = LocalDataSource(args.data).load()
    preprocessed = preprocess(data)
    result = train(preprocessed.train, preprocessed.mappings)
    predictions = predict(result.trace, preprocessed.test)
    predictions.to_csv(args.output_path)


if __name__ == "__main__":
    main()
