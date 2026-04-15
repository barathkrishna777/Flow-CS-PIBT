import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import numpy as np
import pdb

DISCRETE_ACTION_DELTAS = {
    (0, 0): 0,    # wait
    (0, 1): 1,    # right
    (1, 0): 2,    # down
    (-1, 0): 3,   # up
    (0, -1): 4,   # left
}


def discrete_action_labels_from_positions(discrete_positions, t_step):
    """Return exact next-action labels from integer MAPF positions.

    Label mapping is 0=wait, 1=right, 2=down, 3=up, 4=left. The final
    timestep has no next position, so all agents are labeled wait.
    """
    positions = np.asarray(discrete_positions)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(
            "discrete_positions must have shape (num_agents, timesteps, 2)"
        )

    if t_step < 0 or t_step >= positions.shape[1]:
        raise IndexError(f"t_step {t_step} outside trajectory length {positions.shape[1]}")

    if t_step + 1 >= positions.shape[1]:
        deltas = np.zeros((positions.shape[0], 2), dtype=np.int64)
    else:
        deltas = positions[:, t_step + 1, :] - positions[:, t_step, :]
        deltas = np.rint(deltas).astype(np.int64, copy=False)

    labels = np.full(deltas.shape[0], -1, dtype=np.int64)
    for delta, action in DISCRETE_ACTION_DELTAS.items():
        mask = (deltas[:, 0] == delta[0]) & (deltas[:, 1] == delta[1])
        labels[mask] = action

    invalid = np.flatnonzero(labels < 0)
    if invalid.size > 0:
        examples = deltas[invalid[:5]].tolist()
        raise ValueError(f"Unexpected non-cardinal action deltas at timestep {t_step}: {examples}")

    return labels


def create_data_object(pos_list, bd_list, grid, k, m, goal_locs, labels=np.array([]), debug_checks=False):
    """
    pos_list: (N,2) positions
    bd_list: (N,W,H) bd's
    grid: (W,H) grid
    k: (int) local region size
    m: (int) number of closest neighbors to consider
    """
    num_layers = 3 # grid and bd_slices intially
    
        
    num_agents = len(pos_list)
    range_num_agents = np.arange(num_agents)

    ### Numpy advanced indexing to get all agent slices at once
    rowLocs = pos_list[:,0][:, None] # (N)->(N,1), Note doing (N)[:,None] adds an extra dimension
    colLocs = pos_list[:,1][:, None] # (N)->(N,1)
    if debug_checks:
        assert(grid[pos_list[:,0], pos_list[:,1]].all() == 0) # Make sure all agents are on empty space

    x_mesh, y_mesh = np.meshgrid(np.arange(-k,k+1), np.arange(-k,k+1), indexing='ij') # Each is (D,D)
    # Adjust indices to gather slices
    x_mesh = x_mesh[None, :, :] + rowLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    y_mesh = y_mesh[None, :, :] + colLocs[:, None, :] # (1,D,D) + (D,1,D) -> (N,D,D)
    grid_slices = grid[x_mesh, y_mesh] # (N,D,D)
    bd_slices = bd_list[range_num_agents[:,None,None], x_mesh, y_mesh] # (N,D,D)
    N,D = bd_slices.shape[0], bd_slices.shape[1]
    node_features = np.empty((N,num_layers,D,D),dtype=np.float32)
    node_feature_idx = 0
    node_features[:,node_feature_idx] = grid_slices
    node_feature_idx +=1
    node_features[:,node_feature_idx] = bd_slices
    node_feature_idx +=1
    goalRowLocs, goalColLocs= goal_locs[:,0][:, None], goal_locs[:,1][:, None]  # (N,1), (N,1)
    matches = (rowLocs == goalRowLocs) & (colLocs == goalColLocs)
    
    # agent positions
    agent_pos = np.zeros((grid.shape[0], grid.shape[1])) # (W,H)
    agent_pos[rowLocs, colLocs] = 1 # (W,H)
    agent_pos_slices = agent_pos[x_mesh, y_mesh] # (N,D,D)
    node_features[:,node_feature_idx] = agent_pos_slices
    node_feature_idx +=1

    deltas = pos_list[:, None, :] - pos_list[None, :, :] # (N,1,2) - (1,N,2) -> (N,N,2), the difference between each agent
    ## Calculate the distance between each agent, einsum is faster than other options
    dists = np.einsum('ijk,ijk->ij', deltas, deltas, optimize='optimal').astype(float) # (N,N), the L2^2 distance between each agent
    # dists2 = np.linalg.norm(deltas, axis=2, ord=2) # (N,N), the distance between each agent
    # dists3 = np.sum(np.abs(deltas)**2, axis=2) # (N,N), the distance between each agent
    # assert(np.allclose(dists1, dists3)) # Make sure the two distance calculations are the same
    # assert(np.allclose(dists1, np.sqrt(dists2))) # Make sure the two distance calculations are the same

    fov_dist = np.any(np.abs(deltas) > k, axis=2) # (N,N,2)->(N,N) bool for if the agent is within the field of view
    dists[fov_dist] = np.inf # Set the distance to infinity if the agent is out of the field of view
    # O(N) partial sort: select m+1 smallest, then sort only those for deterministic ordering
    partitioned_idx = np.argpartition(dists, m+1, axis=1)[:, 1:m+1]
    neighbor_dists = np.take_along_axis(dists, partitioned_idx, axis=1)
    sort_within = np.argsort(neighbor_dists, axis=1)
    closest_neighbors = np.take_along_axis(partitioned_idx, sort_within, axis=1)
    distance_of_neighbors = dists[range_num_agents[:,None],closest_neighbors] # (N,m)
    
    agent_indices = np.repeat(np.arange(num_agents)[None,:], axis=0, repeats=closest_neighbors.shape[-1]).T # (N,m), each row is 0->num_agents
    neighbors_and_source_idx = np.stack([agent_indices, closest_neighbors]) # (2,N,m), 0 stores source agent, 1 stores neigbhor
    selection = distance_of_neighbors != np.inf # (N,m)
    edge_indices = neighbors_and_source_idx[:, selection] # (2, num_edges), [:,i] corresponds to (source, neighbor)
    edge_features = deltas[edge_indices[0], edge_indices[1]] # (num_edges,2), the difference between each agent
    edge_features = edge_features.astype(np.float32)

    if debug_checks:
        assert(node_features[:,0,k,k].all() == 0) # Make sure all agents are on empty space
        
    bd_pred_arr = None
    linear_dimensions = (grid_slices.shape[1]-2)**2 * num_layers
    # TODO get the best location to go next, just according to the bd
    # NOTE: because we pad all bds with a large number, 
    # we should be able to get the up, down, left and right of each bd without fear of invalid indexing
    # (N, [Stop, Right, Down, Up, Left])
    x_mesh2, y_mesh2 = np.meshgrid(np.arange(-1,1+1), np.arange(-1,1+1), indexing='ij') # assumes k at least 1; getting a 3x3 grid centered at the same place
    x_mesh2 = x_mesh2[None, :, :] + rowLocs[:, None, :] #  -> (N,3,3)
    y_mesh2 = y_mesh2[None, :, :] + colLocs[:, None, :] # -> (N,3,3)
    bd_list = bd_list[np.arange(num_agents)[:,None,None], x_mesh2, y_mesh2] # (N,3,3)
    # set diagonal entries to a big number
    flattened = np.reshape(bd_list, (-1, 9)) # (N,9) # (order (top to bot) left mid right, left mid right, left mid right)
    flattened = flattened[:,[(4,5,7,1,3)]].reshape((-1,5)) # (N,5)

    # Create a boolean array where each element is True if it is the minimum in its row
    min_indices = flattened == flattened.min(axis=1, keepdims=True)
    bd_pred_arr = min_indices.astype(np.float32) # (N, 5) non-unique argmin solution
    linear_dimensions+=5
    # pdb.set_trace()
    
    return Data(x=torch.from_numpy(node_features), edge_index=torch.from_numpy(edge_indices), 
                edge_attr=torch.from_numpy(edge_features), bd_pred=torch.from_numpy(bd_pred_arr), lin_dim=linear_dimensions, num_channels=num_layers,
                y = torch.from_numpy(labels))
    
def get_bd_prefs(pos_list, bds, range_num_agents, add_noise=True):
    """
    pos_list: (N,2) positions
    bds: (N,W,H) bd's
    range_num_agents: (N) range of number of agents
    add_noise: (bool) whether to add noise to break ties
    """
    x_mesh2, y_mesh2 = np.meshgrid(np.arange(-1,1+1), np.arange(-1,1+1), indexing='ij') # assumes k at least 1; getting a 3x3 grid centered at the same place
    x_mesh2 = x_mesh2[None, :, :] + np.expand_dims(pos_list[:,0], axis=(1,2)) #  -> (N,3,3)
    y_mesh2 = y_mesh2[None, :, :] + np.expand_dims(pos_list[:,1], axis=(1,2)) # -> (N,3,3)
    bd_subset = bds[range_num_agents[:,None,None], x_mesh2, y_mesh2] # (N,3,3)
    flattened = np.reshape(bd_subset, (-1, 9)) # (N,9) order (top to bot) left mid right, left mid right, left mid right
    flattened = flattened[:,(4,5,7,1,3)] # (N,5) consistent with NN
    if add_noise:
        # NOTE: Random noise is extremely important for PIBT to work well
        flattened = flattened.astype(float) + np.random.random(flattened.shape)*1e-6 # Add noise to break ties
    else:
        flattened = flattened.astype(float)
    prefs = np.argsort(flattened, axis=1, kind="quicksort") # (N,5) Stop, Right, Down, Up, Left
    return prefs

def normalize_graph_data(data, k, edge_normalize="k", bd_normalize="center"):
    """Modifies data in place"""
    ### Normalize edge attributes
    # data.edge_attr (num_edges,2) the deltas in each direction which can be negative
    assert(edge_normalize in ["k"])
    if edge_normalize == "k":
        data.edge_attr /= k # Normalize edge attributes
    else:
        raise KeyError("Invalid edge normalization method: {}".format(edge_normalize))

    ### Normalize bd
    assert(bd_normalize in ["center"])
    bd_grid = data.x # (N,2,D,D)
    center = bd_grid[:, 1, k, k].unsqueeze(1).unsqueeze(2) # (N,1,1)
    bd_grid[:, 1, :, :] -= center
    bd_grid[:, 1, :, :] *= (1 - bd_grid[:, 0, :, :])
    bd_grid[:, 1, :, :] /= (2*k)
    bd_grid[:, 1, :, :] = torch.clamp(bd_grid[:, 1, :, :], min=-1.0, max=1.0)

    data.x = bd_grid
    assert(data.x[:,1,k,k].all() == 0) # Make sure all agents are on empty space
    assert(data.x[:,1].max() <= 1.0 and data.x[:,1].min() >= -1.0) # Make sure all agents are on empty space
    assert(data.x[:,0,k,k].all() == 0) # Make sure all agents are on empty space
    return data


def load_grid_map_from_file(map_file):
    with open(map_file, 'r') as f:
        f.readline()
        height = int(f.readline().split()[1])
        width = int(f.readline().split()[1])
        f.readline()
        map_data = np.zeros((height, width), dtype=np.int8)
        for r in range(height):
            line = f.readline().strip()
            for c in range(width):
                if line[c] in ['@', 'T', 'O']:
                    map_data[r, c] = 1
    return map_data


def extract_continuous_patches(grid, positions, k):
    """Rasterize local obstacle patches around float positions.

    positions are in map coordinates with cell centers at integer + 0.5.
    """
    positions = np.asarray(positions, dtype=np.float32)
    offsets = np.arange(-k, k + 1, dtype=np.float32)
    row_coords = positions[:, 0][:, None, None] + offsets[None, :, None]
    col_coords = positions[:, 1][:, None, None] + offsets[None, None, :]
    row_idx = np.floor(row_coords).astype(np.int64)
    col_idx = np.floor(col_coords).astype(np.int64)
    row_idx = np.clip(row_idx, 0, grid.shape[0] - 1)
    col_idx = np.clip(col_idx, 0, grid.shape[1] - 1)
    return grid[row_idx, col_idx].astype(np.float32)


def build_continuous_neighbor_graph(pos_list, m, neighbor_radius=None):
    pos_list = np.asarray(pos_list, dtype=np.float32)
    num_agents = len(pos_list)
    if num_agents <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 2), dtype=np.float32)

    deltas = pos_list[:, None, :] - pos_list[None, :, :]
    dists = np.einsum('ijk,ijk->ij', deltas, deltas, optimize='optimal').astype(np.float32)
    np.fill_diagonal(dists, np.inf)

    if neighbor_radius is not None:
        dists[dists > neighbor_radius ** 2] = np.inf

    m_eff = min(max(m, 1), max(num_agents - 1, 1))
    part = np.argpartition(dists, m_eff, axis=1)[:, :m_eff]
    part_dists = np.take_along_axis(dists, part, axis=1)
    order = np.argsort(part_dists, axis=1)
    closest = np.take_along_axis(part, order, axis=1)
    neighbor_dists = dists[np.arange(num_agents)[:, None], closest]
    valid = np.isfinite(neighbor_dists)

    src = np.repeat(np.arange(num_agents)[:, None], closest.shape[1], axis=1)
    edge_indices = np.stack([src, closest])[:, valid]
    edge_features = deltas[edge_indices[0], edge_indices[1]].astype(np.float32)
    return edge_indices, edge_features


def velocity_to_direction_labels(velocities, num_directions=8, wait_threshold=0.1):
    velocities = np.asarray(velocities, dtype=np.float32)
    norms = np.linalg.norm(velocities, axis=1)
    labels = np.zeros(len(velocities), dtype=np.int64)
    moving = norms >= wait_threshold
    if np.any(moving):
        angles = np.arctan2(velocities[moving, 0], velocities[moving, 1])
        bins = np.floor(((angles + np.pi) / (2 * np.pi)) * num_directions).astype(np.int64)
        bins = np.mod(bins, num_directions)
        labels[moving] = bins + 1
    return labels


def labels_to_direction_vectors(labels, num_directions=8):
    labels = np.asarray(labels, dtype=np.int64)
    vecs = np.zeros((len(labels), 2), dtype=np.float32)
    moving = labels > 0
    if np.any(moving):
        angles = ((labels[moving] - 1).astype(np.float32) + 0.5) * (2 * np.pi / num_directions) - np.pi
        vecs[moving, 0] = np.sin(angles)
        vecs[moving, 1] = np.cos(angles)
    return vecs


def create_continuous_data_object(
    pos_list,
    goal_locs,
    grid,
    k,
    m,
    labels=None,
    action_labels=None,
    neighbor_radius=None,
    max_speed=1.0,
):
    pos_list = np.asarray(pos_list, dtype=np.float32)
    goal_locs = np.asarray(goal_locs, dtype=np.float32)
    num_agents = len(pos_list)
    patch_size = 2 * k + 1

    map_patches = extract_continuous_patches(grid, pos_list, k)
    rel_goal = goal_locs - pos_list
    goal_dx = np.broadcast_to(rel_goal[:, 0][:, None, None], (num_agents, patch_size, patch_size)).astype(np.float32)
    goal_dy = np.broadcast_to(rel_goal[:, 1][:, None, None], (num_agents, patch_size, patch_size)).astype(np.float32)

    deltas = pos_list[:, None, :] - pos_list[None, :, :]
    dists = np.sqrt(np.sum(deltas ** 2, axis=2, dtype=np.float32)).astype(np.float32)
    agent_patch = np.zeros((num_agents, patch_size, patch_size), dtype=np.float32)
    offsets = np.arange(-k, k + 1, dtype=np.float32)
    row_grid, col_grid = np.meshgrid(offsets, offsets, indexing='ij')
    for i in range(num_agents):
        rel = pos_list - pos_list[i]
        occupancy = np.zeros_like(agent_patch[i])
        for j in range(num_agents):
            if i == j:
                continue
            sigma = max(0.5, dists[i, j])
            occupancy += np.exp(-((row_grid - rel[j, 0]) ** 2 + (col_grid - rel[j, 1]) ** 2) / (2 * sigma ** 2))
        agent_patch[i] = np.clip(occupancy, 0.0, 1.0)

    node_features = np.stack([map_patches, goal_dx, goal_dy, agent_patch], axis=1).astype(np.float32)
    edge_index, edge_attr = build_continuous_neighbor_graph(pos_list, m, neighbor_radius=neighbor_radius)

    goal_dist = np.linalg.norm(rel_goal, axis=1, keepdims=True)
    at_goal = (goal_dist <= 0.25).astype(np.float32)
    aux_features = np.concatenate(
        [
            np.clip(rel_goal / max(k, 1), -1.0, 1.0),
            np.clip(goal_dist / max(k, 1), 0.0, 1.0),
            at_goal,
            np.full((num_agents, 1), max_speed, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    if labels is None:
        labels = np.zeros((num_agents, 2), dtype=np.float32)
    if action_labels is None:
        action_labels = velocity_to_direction_labels(labels)

    return Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        aux_features=torch.from_numpy(aux_features),
        y=torch.from_numpy(np.asarray(labels, dtype=np.float32)),
        action_label=torch.from_numpy(np.asarray(action_labels, dtype=np.int64)),
    )


def normalize_continuous_graph_data(data, k, max_speed=1.0):
    data.edge_attr = data.edge_attr.float() / max(float(k), 1.0)
    data.edge_attr = torch.clamp(data.edge_attr, min=-1.0, max=1.0)
    data.x = data.x.float()
    data.x[:, 0] = torch.clamp(data.x[:, 0], 0.0, 1.0)
    data.x[:, 1] = torch.clamp(data.x[:, 1] / max(float(k), 1.0), -1.0, 1.0)
    data.x[:, 2] = torch.clamp(data.x[:, 2] / max(float(k), 1.0), -1.0, 1.0)
    data.x[:, 3] = torch.clamp(data.x[:, 3], 0.0, 1.0)
    if hasattr(data, "y") and data.y is not None:
        data.y = data.y.float() / max(float(max_speed), 1.0)
    if hasattr(data, "aux_features") and data.aux_features is not None:
        data.aux_features = data.aux_features.float()
    if hasattr(data, "action_label") and data.action_label is not None:
        data.action_label = data.action_label.long()
    return data
