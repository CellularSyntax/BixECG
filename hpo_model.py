import argparse
import subprocess
import sys

#from src.utils.validation_helper_fns import print_ascii_logo
import src.hpo as module_hpo

def launch_hpo(model_name):
    script_map = {
        "peimankar": ["PeimankarHPO", "conf/peimankarcnnbilstm.json"],
        "bixlstm": ["BiXLSTMHPO", "conf/BiXLSTM1.json"],
        #"jimenez": "src.hpo.hpo_jimenezcnn1d",
        #"liue": "src.hpo.hpo_liuecnnbilstm",
    }

    model_key = model_name.strip().lower()
    if model_key not in script_map:
        print(f"❌ Unknown model: '{model_name}'. Available options: {list(script_map.keys())}")
        sys.exit(1)

    module_to_run, config_path = script_map[model_key]
    print(f"🚀 Launching hyper parameter optimization for model: {model_name}...")

    hpo = getattr(module_hpo, module_to_run)()
    hpo.main(config_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 ECG Model Training Launcher"
    )
    # parser.add_argument("--model", type=str, default="bixlstm",
    #                     help="Model name to train. Options: Peimankar, BiXLSTM, Jimenez, Liue")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to train. Options: Peimankar, BiXLSTM, Jimenez, Liue")
    return parser.parse_args()

def main():
    #print_ascii_logo()
    args = parse_args()
    launch_hpo(args.model)

if __name__ == "__main__":
    main()
