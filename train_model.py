import argparse
import subprocess
import sys

from src.utils.validation_helper_fns import print_ascii_logo

def launch_training(model_name):
    script_map = {
        "peimankar": "src.training.training_peimankarcnnbilstm",
        "bixlstm": "src.training.training_bixlstm",
        "jimenez": "src.training.training_jimenezcnn1d",
        "liue": "src.training.training_liuecnnbilstm",
    }

    model_key = model_name.strip().lower()
    if model_key not in script_map:
        print(f"❌ Unknown model: '{model_name}'. Available options: {list(script_map.keys())}")
        sys.exit(1)

    module_to_run = script_map[model_key]
    print(f"🚀 Launching training for model: {model_name}...")

    try:
        subprocess.run([sys.executable, "-m", module_to_run], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Training script exited with error: {e}")
        sys.exit(e.returncode)

def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 ECG Model Training Launcher"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to train. Options: Peimankar, BiXLSTM, Jimenez, Liue")
    return parser.parse_args()

def main():
    print_ascii_logo()
    args = parse_args()
    launch_training(args.model)

if __name__ == "__main__":
    main()
