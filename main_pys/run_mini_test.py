import subprocess
import os
import numpy as np
import pandas as pd
import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def runBatchExps(mapName: str, num_scens: int, output_csv:str, shieldType: str, lacamLookahead: int = None):
    command = f"python -m main_pys.simple_batch_runner {mapName}"
    command += " --modelPath=data/model/ssil_model.pt --maxSteps=10x --seed=0 --useGPU=False"
    command += f" --shieldType={shieldType}"
    if lacamLookahead is not None:
        command += f" --lacamLookahead={lacamLookahead}"
    command += f" --num_scens={num_scens} --outputCSV={output_csv}"
    
    subprocess.run(command, shell=True, check=True)

"""
python -m main_pys.run_mini_test --mapName=den312d
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mini test")
    parser.add_argument("--mapName", type=str, help="Map name", required=True)
    args = parser.parse_args()

    # Run the batch experiments
    num_scens = 10
    output_csv = f"{args.mapName}.csv"
    runBatchExps(args.mapName, num_scens, output_csv, "CS-Freeze")
    runBatchExps(args.mapName, num_scens, output_csv, "CS-PIBT")
    runBatchExps(args.mapName, num_scens, output_csv, "Real-Time-LaCAM", lacamLookahead=1)
    
    # Create a plot
    FONT_SIZE = 10
    plt.rcParams.update({'font.size': FONT_SIZE})  # Increase base font size
    
    data = pd.read_csv(f"logs/{output_csv}")
    data = data[['agentNum', 'shieldType', 'success']]
    for shield in data['shieldType'].unique():
        shield_data = data[data['shieldType'] == shield]
        success_rate = shield_data.groupby('agentNum')['success'].mean()
        if shield == "Real-Time-LaCAM":
            shield = "SSIL + Real-Time LaCAM"
        if shield == "CS-PIBT":
            shield = "SSIL + CS-PIBT"
        plt.plot(success_rate.index, success_rate, marker="o", markersize=8, label=shield)
    
    plt.ylim(0, 1.05)
    plt.xlabel('Number of Agents', fontsize=FONT_SIZE + 2)
    plt.ylabel('Success Rate', fontsize=FONT_SIZE + 2)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.legend(fontsize=FONT_SIZE, frameon=True)
    # plt.title(f"Effect of different collision shields on {args.mapName}", fontsize=TITLE_SIZE)
        
    plt.savefig(f"logs/{args.mapName}_success_rate.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Results saved to logs/{output_csv}")
    print(f"Plot saved to logs/{args.mapName}_success_rate.png")
    