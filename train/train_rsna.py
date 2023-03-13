import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="configs")
    parser.add_argument('--config', type=str, help='Path to json config file', default="train/train_rsna_config.json")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config_dict = json.load(f)


if __name__ == '__main__':
    main()