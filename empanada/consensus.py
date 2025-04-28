import numpy as np
import networkx as nx
from itertools import combinations
from tqdm import tqdm
from empanada.array_utils import *

MIN_OVERLAP = 100
MIN_IOU = 1e-2

def average_edge_between_clusters(G, cluster1, cluster2, key='iou'):
    r"""Calculates the average edge weight between two groups of nodes
    in a graph.

    Args:
        G: nx.Graph containing the nodes in cluster1 and cluster2
        cluster1: List of nodes in G
        cluster2: List of nodes in G
        key: Name of the edge weight.

    Returns:
        avg_weight: Float, the average edge weight across clusters.

    """
    weights = []

    # get pairwise edge weights
    for node1 in cluster1:
        for node2 in cluster2:
            weights.append(
                G[node1][node2][key] if G.has_edge(node1, node2) else 0
            )

    return sum(weights) / len(weights)

def create_graph_of_clusters(G, cluster_iou_thr):
    r"""Creates a graph in which each node is a group
    of nodes that have IoU greater than cluster_iou_thr.

    Args:
        G: nx.Graph containing detection nodes
        cluster_iou_thr: Minimum IoU score between nodes for them
            to be put into the same group.

    Returns:
        cluster_graph: nx.Graph containing the grouped detection nodes.
        Nodes are groups of detections and edges denote groups that have
        overlap with each other.

    """
    # create new graph with low iou edges dropped
    H = G.copy()
    for (u, v, d) in G.edges(data=True):
        if d['iou'] <= cluster_iou_thr:
            H.remove_edge(u, v)

    # each cluster is a connected component in the new graph
    cluster_graph = nx.Graph()
    for i,cluster in enumerate(nx.connected_components(H)):
        cluster_graph.add_node(i, cluster=cluster)

    # add edges weighted by average edge weight
    # in the non-cluster graph
    for node1,node2 in combinations(cluster_graph.nodes, 2):
        cluster1 = cluster_graph.nodes[node1]['cluster']
        cluster2 = cluster_graph.nodes[node2]['cluster']

        # get edge weights
        iou_weight = average_edge_between_clusters(G, cluster1, cluster2, 'iou')
        overlap_weight = average_edge_between_clusters(G, cluster1, cluster2, 'overlap')

        if iou_weight > MIN_IOU or overlap_weight > MIN_OVERLAP:
            cluster_graph.add_edge(node1, node2, iou=iou_weight, overlap=overlap_weight)

    return cluster_graph

def push_cluster(G, src, dst):
    r"""
    Merges groups from two nodes in a cluster_graph and removes their edge.
    """
    src_cluster = G.nodes[src]['cluster']
    G.nodes[dst]['cluster'] = G.nodes[dst]['cluster'].union(src_cluster)
    G.remove_edge(src, dst)

    return G

def merge_clusters(G):
    r"""Merges together clusters in the cluster graph iteratively.

    Args:
        G: nx.Graph containing nodes that represent groups of detections.

    Returns:
        H: nx.Graph containing nodes that represent the merged groups
        of detections.

    """
    # copy to avoid inplace changes
    H = G.copy()

    while len(H.edges()) > 0:
        # most connected from sorted nodes by the number of neighbors
        most_connected = sorted(
            H.nodes, key=lambda x: len(list(H.neighbors(x))), reverse=True
        )[0]

        # sort neighbors by the size of their clusters
        neighbors = sorted(
            H.neighbors(most_connected),
            key=lambda x: len(H.nodes[x]['cluster']),
            reverse=True
        )

        # decide whether to push the most connected cluster to
        # merge with its neighbors or to merge all the neighbors
        # into the most connected cluster
        most_connected_cluster = H.nodes[most_connected]['cluster']

        # if a neighbor has a bigger cluster then push most connected
        push_most_connected = len(H.nodes[neighbors[0]]['cluster']) > len(most_connected_cluster)

        if push_most_connected:
            # most connected cluster is rejected as an instance
            for neighbor in neighbors:
                push_cluster(H, most_connected, neighbor)

            H.remove_node(most_connected)
        else:
            # most connected cluster is accepted as an instance
            # pull all the neighboring clusters
            for neighbor in neighbors:
                push_cluster(H, neighbor, most_connected)

                # push secondary neighbors to most connected node
                second_neighbors = list(H.neighbors(neighbor))
                for sn in second_neighbors:
                    if not H.has_edge(most_connected, sn):
                        edge_iou = H[neighbor][sn]['iou']
                        H.add_edge(most_connected, neighbor, iou=edge_iou)

                H.remove_node(neighbor)

    return H

def merge_instances(instances_dict):
    r"""Merge arbitrary number of instances. From dict
    of instance_id and instance_attrs.
    """
    if len(instances_dict) < 2:
        return list(instances_dict.values())[0]

    merged_box, merged_starts, merged_runs = None, None, None
    for instance_attrs in instances_dict.values():
        if merged_box is None:
            merged_box = instance_attrs['box']
            merged_starts = instance_attrs['starts']
            merged_runs = instance_attrs['runs']
        else:
            merged_box = merge_boxes(merged_box, instance_attrs['box'])
            merged_starts, merged_runs = merge_rles(
                merged_starts, merged_runs,
                instance_attrs['starts'], instance_attrs['runs']
            )

    return dict(box=merged_box, starts=merged_starts, runs=merged_runs)

def merge_overlapping(cluster_instances):
    r"""Merges together instances that have non-trivial overlap with
    each other.
    """
    # only applies when more than 1 instance in a cluster
    if len(cluster_instances) < 2:
        return list(cluster_instances.values())

    # resolve overlaps between cluster instances
    instance_ids = list(cluster_instances.keys())
    merge_graph = nx.Graph()
    merge_graph.add_nodes_from(instance_ids)

    # measure intersection between all pairs of instances
    for c_i,c_j in combinations(instance_ids, 2):
        pair_iou, inter_area = rle_iou(
            cluster_instances[c_i]['starts'], cluster_instances[c_i]['runs'],
            cluster_instances[c_j]['starts'], cluster_instances[c_j]['runs'],
            return_intersection=True
        )

        if pair_iou > MIN_IOU or inter_area > MIN_OVERLAP:
            merge_graph.add_edge(c_i, c_j)

    merged_instances = []
    for comp in nx.connected_components(merge_graph):
        comp_instances = {k: v for k,v in cluster_instances.items() if k in comp}
        merged_instances.append(merge_instances(comp_instances))

    return merged_instances

def bounding_box_screening(boxes, source_indices):
    r"""Merges together clusters in the cluster graph iteratively.

    Args:
        boxes: Array of size (n, 4) or (n, 6) where bounding box
            is defined as (y1, x1, y2, x2) or (z1, y1, x1, z2, y2, x2).

        source_indices: Array of size (n,) that records the source of each
            bounding box. Bounding boxes from the same source are always screened.

    Returns:
        box_matches: Array of size (k, 2). Each item is a unique pair of bounding
        boxes from boxes that have non-trivial overlap with each other.

    """

    # compute pairwise overlaps for all distance boxes
    # TODO: replace pairwise intersection calculation with something
    # more memory efficient (only matters when N is large ~10^4)
    box_ious = box_iou(boxes)
    
    # use small value to weed out really trivial overlaps
    box_matches = np.array(box_ious.nonzero()).T

    # filter out boxes from the same source (mask or tracker)
    r1_match_tr = source_indices[box_matches[:, 0]]
    r2_match_tr = source_indices[box_matches[:, 1]]
    box_matches = box_matches[r1_match_tr != r2_match_tr]

    # order of items in pair doesn't matter,
    # remove duplicates from symmetric matrix
    box_matches = np.sort(box_matches, axis=-1)
    box_matches = np.unique(box_matches, axis=0)

    return box_matches

def object_iou_graph(
    source_indices,
    object_labels, 
    object_boxes, 
    object_starts, 
    object_runs
):
    r"""Creates a graph where each node is an object and each edge
    represents non-zero overlaps between objects.

    Args:
        source_indices: Array of size (n,) that records the source of each object.
            E.g., The index of tiles cropped from a larger image.

        objects_labels: Array of size (n,) that records the label of each object.

        object_boxes: Array of size (n, 4) or (n, 6) where bounding box
            is defined as (y1, x1, y2, x2) or (z1, y1, x1, z2, y2, x2).

        object_starts: Array of size (n,) that records the start index of each
            object run length encoding.

        object_starts: Array of size (n,) that records the run length of each
            object run length encoding.

    Returns:
        object_graph: NetworkX graph where each node is an object and each edge
            represents non-zero overlaps between objects.
    
    """
    # screen possible matches by bounding box first
    box_matches = bounding_box_screening(object_boxes, source_indices)

    # create graph with nodes
    graph = nx.Graph()
    for node_id in range(len(object_labels)):
        graph.add_node(
            node_id, box=object_boxes[node_id],
            starts=object_starts[node_id],
            runs=object_runs[node_id]
        )

    # iou as weighted edges
    for r1, r2 in zip(*tuple(box_matches.T)):
        pair_iou, inter_area = rle_iou(
            graph.nodes[r1]['starts'], graph.nodes[r1]['runs'],
            graph.nodes[r2]['starts'], graph.nodes[r2]['runs'],
            return_intersection=True
        )

        # add edge for non-trivial overlaps
        if pair_iou > 0:
            graph.add_edge(r1, r2, iou=pair_iou, overlap=inter_area)

    return graph

def merge_semantic_from_trackers(
    semantic_trackers,
    pixel_vote_thr=2
):
    r"""Performs the consensus creation algorithm for instances from an
    arbitrary number of trackers (see empanada.inference.trackers).

    Args:
        semantic_trackers: List of empanada.inference.InstanceTracker. There
        should only be a single instance for a semantic class.

        pixel_vote_thr: Integer. Number of votes for a pixel/voxel to
        be in the consensus segmentation. Default 2, assumes there are
        3 semantic trackers.

    Returns:
        instances: A nested. dictionary of instances. Each key is an instance_id.
        Values are themselves dictionaries that contain the bounding box
        and run length encoding of the instance ('boxes', 'starts', 'runs').
        For semantic seg there is only a single instance id: 1.

    """
    
    # extract the run length encoded segmentations
    boxes = []
    starts = []
    runs = []
    for tr in semantic_trackers:
        assert len(tr.instances.keys()) <= 1, 'Semantic classes only have 1 label!'
        for attrs in tr.instances.values():
            boxes.append(attrs['box'])
            starts.append(attrs['starts'])
            runs.append(attrs['runs'])
          
    # no segs found
    if not boxes:
        return {}
            
    # merge the boxes
    merged_box = boxes[0]
    for box in boxes[1:]:
        merged_box = merge_boxes(merged_box, box)
            
    # concat rles to ranges
    seg_ranges = [
        np.stack([s, s + r], axis=1) for s,r in zip(starts, runs)
    ]
    
    # sort the ranges and vote on pixels
    seg_ranges = vote_by_ranges(seg_ranges, pixel_vote_thr)
    
    seg_attrs = {
        'box': merged_box, 'starts': seg_ranges[:, 0],
        'runs': seg_ranges[:, 1] - seg_ranges[:, 0]
    }
    
    # value of semantic class is 1 now
    return {1: seg_attrs}

def merge_objects_from_trackers(
    object_trackers,
    pixel_vote_thr=2,
    cluster_iou_thr=0.75,
    bypass=False
):
    r"""Performs the consensus creation algorithm for instances from an
    arbitrary number of trackers (see empanada.inference.trackers).

    Args:
        object_trackers: List of empanada.inference.InstanceTracker

        pixel_vote_thr: Integer. Number of votes for a pixel/voxel to
            be in the consensus segmentation. Default 2, assumes there are
            3 object trackers.

        cluster_iou_thr: Float. IoU threshold for merging groups of instances.
            Default 0.75.

        bypass: Bool. If True, instances that appear in just 1 of the object
            trackers can be included in the consensus. This will only affect the
            final segmentation if pixel_vote_thr < 0.5 * len(object_trackers).
            Default False.

    Returns:
        instances: A nested dictionary of instances. Each key is an instance_id.
        Values are themselves dictionaries that contain the bounding box
        and run length encoding of the instance ('boxes', 'starts', 'runs').

    """
    n_votes = len(object_trackers)

    if bypass:
        min_cluster_size = 1
    else:
        # better to require majority clusters
        # even when not majority voxels
        min_cluster_size = (n_votes // 2) + 1

    # default to maximal merging when
    # not using majority vote
    if pixel_vote_thr < min_cluster_size:
        cluster_iou_thr = 0

    # unpack the instances from each tracker
    # into arrays for labels, bounding boxes
    # and voxel locations
    tracker_indices = []
    object_labels = []
    object_boxes = []
    object_starts = []
    object_runs = []
    for tr_index, tr in enumerate(object_trackers):
        for instance_id, instance_attr in tr.instances.items():
            tracker_indices.append(tr_index)
            object_labels.append(int(instance_id))
            object_boxes.append(instance_attr['box'])
            object_starts.append(instance_attr['starts'])
            object_runs.append(instance_attr['runs'])

    # store in arrays for convenient slicing
    tracker_indices = np.array(tracker_indices)
    object_labels = np.array(object_labels)
    object_boxes = np.array(object_boxes)

    if len(object_boxes) == 0:
        # no instances to return
        return {}

    # create a graph with a node for each object
    # and edges that connected objects whose 
    # masks have non-zero overlap
    graph = object_iou_graph(
        tracker_indices, object_labels, object_boxes, 
        object_starts, object_runs
    )

    instance_id = 1
    instances = {}
    for comp in nx.connected_components(graph):
        if len(comp) < min_cluster_size:
            continue

        cluster_graph = create_graph_of_clusters(graph.subgraph(comp), cluster_iou_thr)
        cluster_graph = merge_clusters(cluster_graph)

        cluster_id = 1
        cluster_instances = {}
        for node in cluster_graph.nodes:
            cluster = list(cluster_graph.nodes[node]['cluster'])

            if len(cluster) < min_cluster_size:
                continue

            # merge boxes and coords from nodes
            node0 = cluster[0]
            merged_box = graph.nodes[node0]['box']
            for node_id in cluster[1:]:
                merged_box = merge_boxes(merged_box, graph.nodes[node_id]['box'])

            # vote on indices that should belong to an object
            all_ranges = [
                np.stack([graph.nodes[node_id]['starts'], graph.nodes[node_id]['starts'] + graph.nodes[node_id]['runs']], axis=1)
                for node_id in cluster
            ]
            voted_ranges = vote_by_ranges(all_ranges, pixel_vote_thr) 

            if len(voted_ranges) > 0:
                cluster_instances[cluster_id] = {}
                cluster_instances[cluster_id]['box'] = tuple(map(lambda x: x.item(), merged_box))

                cluster_instances[cluster_id]['starts'] = voted_ranges[:, 0]
                cluster_instances[cluster_id]['runs'] = voted_ranges[:, 1] - voted_ranges[:, 0]

                cluster_id += 1

        # merge together instances with higher than trivial overlap
        for instance_attrs in merge_overlapping(cluster_instances):
            instances[instance_id] = instance_attrs
            instance_id += 1

    return instances

def merge_semantic_from_tiles(tiles):
    r"""Merges run length encoded semantic segmentations
    from a list of tiles.

    Args:
        tiles: RLE segmentation of a tile. Segmentations are expected
            to be for a single instance or semantic class.

    Returns:
        merged_rles: The merged RLE segmentation for the tiles.

    """
    # unpack the instances from each tracker
    # into arrays for labels, bounding boxes
    # and voxel locations
    label_id = None
    boxes = []
    starts = []
    runs = []
    for tile_instances in tiles:
        for instance_id, instance_attr in tile_instances.items():
            if label_id is None:
                label_id = instance_id

            boxes.append(instance_attr['box'])
            starts.append(instance_attr['starts'])
            runs.append(instance_attr['runs'])

    boxes = np.array(boxes)

    if len(boxes) == 0:
        # no instances to return
        return {}

    # merge the boxes
    merged_box = boxes[0]
    for box in boxes[1:]:
        merged_box = merge_boxes(merged_box, box)
            
    # concat rles to ranges
    seg_ranges = [
        np.stack([s, s + r], axis=1) for s,r in zip(starts, runs)
    ]
    
    # sort the ranges and vote on pixels
    seg_ranges = join_ranges(seg_ranges)
    
    seg_attrs = {
        'box': merged_box, 'starts': seg_ranges[:, 0],
        'runs': seg_ranges[:, 1] - seg_ranges[:, 0]
    }
    
    # value of semantic class is 1 now
    return {label_id: seg_attrs}

def merge_objects_from_tiles(tiles, overlap_rle=None):
    r"""Merges run length encoded instance or semantic segmentations
    from a list of tiles.

    Args:
        tiles: RLE segmentation of a tile. Segmentations are expected
            to be for a single instance or semantic class.

        overlap_rle: RLE of the overlapping regions between tiles.
            E.g., The overlap_rle created by a tiler (see Tiler.overlap_rle).
            If given, objects detected in the overlapping region are required
            to be detected in at least 2 tiles. Useful for filtering out
            false positives but takes longer to compute.

    Returns:
        merged_rles: The merged RLE segmentation for the tiles.

    """
    # unpack the instances from each tracker
    # into arrays for labels, bounding boxes
    # and voxel locations
    tile_indices = []
    object_labels = []
    object_boxes = []
    object_starts = []
    object_runs = []
    for tile_idx, tile_instances in enumerate(tiles):
        for instance_id, instance_attr in tile_instances.items():
            tile_indices.append(tile_idx)
            object_labels.append(int(instance_id))
            object_boxes.append(instance_attr['box'])
            object_starts.append(instance_attr['starts'])
            object_runs.append(instance_attr['runs'])

    # store in arrays for convenient slicing
    tile_indices = np.array(tile_indices)
    object_labels = np.array(object_labels)
    object_boxes = np.array(object_boxes)

    if len(object_boxes) == 0:
        # no instances to return
        return {}

    # create a graph with a node for each object
    # and edges that connected objects whose 
    # masks have non-zero overlap
    graph = object_iou_graph(
        tile_indices, object_labels, object_boxes, 
        object_starts, object_runs
    )

    if overlap_rle is not None:
        overlap_starts, overlap_runs = overlap_rle

    instance_id = int(np.min(object_labels))
    instances = {}
    for cluster in nx.connected_components(graph):
        cluster = list(cluster)
        
        # merge boxes and coords from nodes
        node0 = cluster[0]
        merged_box = graph.nodes[node0]['box']
        for node_id in cluster[1:]:
            merged_box = merge_boxes(merged_box, graph.nodes[node_id]['box'])

        # vote on indices that should belong to an object
        all_ranges = [
            np.stack([graph.nodes[node_id]['starts'], graph.nodes[node_id]['starts'] + graph.nodes[node_id]['runs']], axis=1)
            for node_id in cluster
        ]

        # vote thr of 2 works within overlapping tiles
        # joined ranges works otherwise
        #voted_ov_ranges = np.array(rle_voting(all_ranges, 2))
        voted_ranges = join_ranges(all_ranges)
        if overlap_rle is not None and len(cluster) < 2 and np.any(voted_ranges):
            voted_rle = ranges_to_rle(voted_ranges)

            # check whether cluster is partially contained
            # in the overlap region
            ov_ioa = rle_ioa(
                overlap_starts, overlap_runs, 
                voted_rle[:, 0], voted_rle[:, 1]
            )

            # if greater than 10% of the object is
            # contained in the overlap region, but
            # has only 1 vote, then it's likely an FP
            if ov_ioa > 0.1:
                voted_ranges = []
        
        if np.any(voted_ranges):
            instances[instance_id] = {}
            instances[instance_id]['box'] = tuple(map(lambda x: x.item(), merged_box))

            instances[instance_id]['starts'] = voted_ranges[:, 0]
            instances[instance_id]['runs'] = voted_ranges[:, 1] - voted_ranges[:, 0]

            instance_id += 1

    return instances