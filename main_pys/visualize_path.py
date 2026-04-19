import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import pdb
import tqdm
import shutil
import tempfile

'''
This script animates the paths of agents in the map.
'''

def create_gif(image_folder, output_path, duration=100, end_frame_duration=2000, cleanup=True):
    images: list[Image.Image] = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.endswith(('png')):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))

    if images:
        duration = [duration] * (len(images) - 1) + [end_frame_duration] # Make last frame longer
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    else:
        print("No images found in the folder")

    if cleanup:
        image_names = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        for image_name in image_names:
            image_path = os.path.join(image_folder, image_name)
            os.remove(image_path)


def parse_scene(scen_file):
    """Input: scenfile
    Output: start_locations, goal_locations
    """
    start_locations = []
    goal_locations = []

    with open(scen_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('version'):
                continue
            tokens = line.split("\t")
            # pdb.set_trace()
            assert(len(tokens) == 9)
            tokens = tokens[4:]
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row, col))
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row, col))
    return np.array(start_locations, dtype=int), np.array(goal_locations, dtype=int)

def readMap(mapfile: str):
    """ Read map """
    if mapfile.startswith("../data"):
        mapfile = mapfile[3:]
    with open(mapfile) as f:
        line = f.readline()
        line = f.readline()
        height = int(line.split(' ')[1])
        line = f.readline()
        width = int(line.split(' ')[1])
        line = f.readline()
        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata.reshape((width, height))
    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    return mapdata


def createAnimation(args):
    mapName = args.mapName
    mapdata = readMap(f"{args.mapFolder}/{mapName}.map")
    id2plan = np.load(args.pathsNpyFilePath) # (T,N,2)
    max_plan_length = id2plan.shape[0]
    num_agents = id2plan.shape[1]
    # pdb.set_trace()
    if args.scenName is not None:
        scen_abbr = args.scenName.split(".")[0]
        scen_file = f"{args.sceneFile}/{scen_abbr}.scen"
        start_locs, id2goal = parse_scene(scen_file)
        success = np.all(id2plan[-1] == id2goal[0:num_agents])
        if success:
            textColor = 'green'
        else:
            textColor = 'red'
    else:
        success = None
        textColor = 'black'
        id2goal = None
    outputFilePath = args.outputGif

    tmpFolder = args.tmpFolderToSaveImages
    if tmpFolder is None:
        tmpFolder = tempfile.mkdtemp(prefix="flow_cs_pibt_frames_")
        cleanup_tmp_dir = True
    else:
        cleanup_tmp_dir = False
    os.makedirs(tmpFolder, exist_ok=True)

    cmap = plt.get_cmap(args.agentCmap, num_agents)
    frames = list(range(0, max_plan_length, max(args.frameStride, 1)))
    if frames[-1] != max_plan_length - 1:
        frames.append(max_plan_length - 1)

    fig, ax = plt.subplots(figsize=(args.figureSize, args.figureSize))
    for frame_idx, t in enumerate(tqdm.tqdm(frames, desc="Creating visualization")):
        ax.clear()
        ax.imshow(mapdata, cmap="Greys", origin="upper", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, mapdata.shape[1] - 0.5)
        ax.set_ylim(mapdata.shape[0] - 0.5, -0.5)

        if id2goal is not None:
            goals = id2goal[:num_agents]
            ax.scatter(
                goals[:, 1],
                goals[:, 0],
                s=args.goalSize,
                marker="*",
                c=[cmap(i) for i in range(num_agents)],
                edgecolors="black",
                linewidths=args.goalEdgeWidth,
                alpha=0.8,
                zorder=3,
            )

        for i in range(num_agents):
            plan = id2plan[:, i]
            color = cmap(i)
            trail_start = max(0, t - args.trailLength)
            if t > trail_start:
                trail = plan[trail_start : t + 1]
                ax.plot(
                    trail[:, 1],
                    trail[:, 0],
                    linewidth=args.trailWidth,
                    c=color,
                    alpha=0.35,
                    zorder=2,
                )
            at_goal = id2goal is not None and np.all(plan[t] == id2goal[i])
            ax.scatter(
                plan[t][1],
                plan[t][0],
                s=args.agentSize,
                c=[color if not at_goal else "lightgrey"],
                edgecolors="black",
                linewidths=args.agentEdgeWidth,
                zorder=4,
            )
            if args.labelAgents and num_agents <= args.maxLabeledAgents:
                ax.text(
                    plan[t][1],
                    plan[t][0],
                    str(i),
                    fontsize=5,
                    ha="center",
                    va="center",
                    zorder=5,
                )
        fig.subplots_adjust(top=0.88)
        name = "{}/{:03d}.png".format(tmpFolder, t)
        ax.set_title(f"{mapName}: t = {t} / {max_plan_length - 1}", color=textColor)
        fig.savefig(name, dpi=args.dpi)
    plt.close(fig)

    os.makedirs(os.path.dirname(outputFilePath) or ".", exist_ok=True)
    create_gif(
        tmpFolder,
        outputFilePath,
        duration=args.frameDurationMs,
        end_frame_duration=args.endFrameDurationMs,
        cleanup=not cleanup_tmp_dir,
    )
    if cleanup_tmp_dir:
        shutil.rmtree(tmpFolder, ignore_errors=True)


"""
Example usage:
python -m main_pys.visualize_path den312d logs/paths.npy --scenName=den312d-random-1.scen 
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize agent paths from log file')
    parser.add_argument("mapName", help="map name without .map, needs to be in mapsToMaxNumAgents defined in the top", type=str) # Note: Positional is required
    parser.add_argument("pathsNpyFilePath", help="Path to the paths.npy file", type=str) # Note: Positional is required
    parser.add_argument("--scenName", help="scen name with .scen", type=str, default=None) # Note: Positional is required
    parser.add_argument("--mapFolder", help="folder containing maps", type=str, default="data/mapf-map")
    parser.add_argument("--sceneFile", help="folder containing maps", type=str, default="data/mapf-scen-random")
    parser.add_argument('--tmpFolderToSaveImages', type=str, help='temporary folder to save images', default=None)
    parser.add_argument('--outputGif', type=str, help='Path to the output gif file', default="logs/paths.gif")
    parser.add_argument('--frameStride', type=int, default=2, help='Render every Nth timestep')
    parser.add_argument('--trailLength', type=int, default=25, help='Number of previous timesteps to draw as trails')
    parser.add_argument('--agentSize', type=float, default=14.0)
    parser.add_argument('--goalSize', type=float, default=28.0)
    parser.add_argument('--trailWidth', type=float, default=0.7)
    parser.add_argument('--agentEdgeWidth', type=float, default=0.2)
    parser.add_argument('--goalEdgeWidth', type=float, default=0.25)
    parser.add_argument('--figureSize', type=float, default=7.0)
    parser.add_argument('--dpi', type=int, default=120)
    parser.add_argument('--frameDurationMs', type=int, default=80)
    parser.add_argument('--endFrameDurationMs', type=int, default=1600)
    parser.add_argument('--agentCmap', type=str, default='turbo')
    parser.add_argument('--labelAgents', action='store_true')
    parser.add_argument('--maxLabeledAgents', type=int, default=30)
    args = parser.parse_args()
    createAnimation(args)
