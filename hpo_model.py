import argparse
import subprocess
import sys
import os
#from src.utils.validation_helper_fns import print_ascii_logo
import src.hpo as module_hpo

def launch_hpo(args):
    script_map = {
        "peimankar": ["PeimankarHPO", "conf/peimankarcnnbilstm.json"],
        "bixlstm": ["BiXLSTMHPO", "conf/BiXLSTM2.json"],
        "bixlstm_no_pad": ["BiXLSTMHPO", "conf/BiXLSTM2_no_pad.json"],
        "bixlstm_nonshared": ["BiXLSTMHPO", "conf/BiXLSTM2_nonshared.json"],
        #"sfxlstm": ["SFXLSTMHPO", "conf/SFXLSTM.json"],
        #"bisfxlstm": ["BiSFXLSTMHPO", "conf/BiSFXLSTM.json"],
        #"jimenez": "src.hpo.hpo_jimenezcnn1d",
        #"liue": "src.hpo.hpo_liuecnnbilstm",
    }

    model_name = args.model
    model_key = model_name.strip().lower()
    if model_key not in script_map:
        print(f"❌ Unknown model: '{model_name}'. Available options: {list(script_map.keys())}")
        sys.exit(1)

    module_to_run, config_path = script_map[model_key]
    print(f"🚀 Launching hyper parameter optimization for model: {model_name}...")

    hpo = getattr(module_hpo, module_to_run)(
        multi_objective=args.multiobj, 
        save_to=args.save_to, 
        hpo_name=args.hpo_name, 
        seed=args.seed,
        beat_aligned=args.beat_aligned)
    hpo.main(config_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 ECG Model Training Launcher"
    )
    
    parser.add_argument("--model", type=str, default="bixlstm",
                        help="Model name to train. Options: Peimankar, BiXLSTM, Jimenez, Liue")
    # parser.add_argument("--model", type=str, required=True,
    #                     help="Model name to train. Options: Peimankar, BiXLSTM, Jimenez, Liue")
    parser.add_argument('--multiobj', default=True, action=argparse.BooleanOptionalAction,
                        help="Start multi-objective optimization (max: macrof1, min: num_params). Default: True")
    parser.add_argument('--save_to', default="results", type=str,
                        help="Path to save the results of the HPO run. Default: results")
    parser.add_argument('--hpo_name', default="DEBUG_hpo", type=str,
                        help="Name of the HPO run. Results will be saved to save_to/hpo_name. Default: hpo")
    parser.add_argument('--seed', default=42, type=int,
                        help="Random seed for reproducibility. Default: 42")
    parser.add_argument("--beat_aligned", dest="beat_aligned", default=True, action="store_true",
                        help="Use beat-aligned ECG segmentation instead of fixed-size windows.")
    parser.add_argument("--no-beat_aligned", dest="beat_aligned", action="store_false",
                        help="")
    return parser.parse_args()

def main():
    #print_ascii_logo()
    args = parse_args()
    launch_hpo(args)

if __name__ == "__main__":
    main()
