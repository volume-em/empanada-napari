import json
import math
import numpy as np
from copy import deepcopy
from empanada.array_utils import *

__all__ = [
    'InstanceTracker'
]

def to_box3d(index2d, box, axis):
    assert axis in ['xy', 'xz', 'yz']

    # extract box ranges
    h1, w1, h2, w2 = box
    if axis == 'xy':
        box3d = (index2d, h1, w1, index2d+1, h2, w2)
    elif axis == 'xz':
        box3d = (h1, index2d, w1, h2, index2d+1, w2)
    else:
        box3d = (h1, w1, index2d, h2, w2, index2d+1)

    return box3d

def to_coords3d(index2d, coords, axis):
    assert axis in ['xy', 'xz', 'yz']

    # split coords into arrays
    hcoords, wcoords = coords #tuple(coords.T)
    dcoords = np.repeat([index2d], len(hcoords))
    if axis == 'xy':
        coords3d = (dcoords, hcoords, wcoords)
    elif axis == 'xz':
        coords3d = (hcoords, dcoords, wcoords)
    else:
        coords3d = (hcoords, wcoords, dcoords)

    return coords3d

class InstanceTracker:
    def __init__(
        self,
        class_id=None,
        label_divisor=None,
        shape3d=None,
        axis='xy'
    ):
        assert axis in ['xy', 'xz', 'yz']
        self.class_id = class_id
        self.label_divisor = label_divisor
        self.shape3d = shape3d
        self.axis = axis
        self.finished = False
        self.reset()

        self.axis_nums = {'xy': 0, 'xz': 1, 'yz': 2}

    def reset(self):
        self.instances = {}

    def update(self, instance_rles, index2d):
        assert self.class_id is not None
        assert self.label_divisor is not None
        assert self.shape3d is not None
        assert not self.finished, "Cannot update tracker after calling finish!"

        # extract bounding box and object pixels coords
        for label,attrs in instance_rles.items():
            box = to_box3d(index2d, attrs['box'], self.axis)
            ignore_idx = self.axis_nums[self.axis]

            shape2d = tuple([s for i,s in enumerate(self.shape3d) if i != ignore_idx])

            # convert to 3d starts and runs
            if self.axis == 'xy':
                starts = attrs['starts'] + index2d * math.prod(shape2d)
                runs = attrs['runs']
            elif self.axis == 'xz':
                coords2d = np.unravel_index(attrs['starts'], shape2d)
                coords3d = to_coords3d(index2d, coords2d, 'xz')
                starts = np.ravel_multi_index(coords3d, self.shape3d)
                runs = attrs['runs']
            else:
                coords2d_flat = rle_decode(attrs['starts'], attrs['runs'])
                coords2d = np.unravel_index(coords2d_flat, shape2d)
                coords3d = to_coords3d(index2d, coords2d, 'yz')
                starts = np.ravel_multi_index(coords3d, self.shape3d)
                runs = np.ones_like(starts)

            # update instances dict
            if label not in self.instances:
                self.instances[label] = {
                    'box': box, 'starts': [starts], 'runs': [runs]
                }
            else:
                # merge boxes and coords
                instance_dict = self.instances[label]
                self.instances[label]['box'] = merge_boxes(box, instance_dict['box'])
                self.instances[label]['starts'].append(starts)
                self.instances[label]['runs'].append(runs)

    def finish(self):
        # concat all starts and runs
        for instance_id in self.instances.keys():
            if isinstance(self.instances[instance_id]['starts'] , list):
                starts = np.concatenate(self.instances[instance_id]['starts'])

                # 3D yz wasn't actually run length encoded
                # because RLE happens along x axis,
                # re-encoding makes calculations faster later
                if self.axis == 'yz':
                    starts, runs = rle_encode(np.sort(starts, kind='stable'))
                else:
                    runs = np.concatenate(self.instances[instance_id]['runs'])

                self.instances[instance_id]['starts'] = starts
                self.instances[instance_id]['runs'] = runs
            else:
                # if starts/runs are not lists, then
                # they've already been concatenated
                continue

        self.finished = True

    def write_to_json(self, savepath):
        if not self.finished:
            self.finish()

        save_dict = deepcopy(self.__dict__)
        # convert instance coords to string
        for k in save_dict['instances'].keys():
            starts = save_dict['instances'][k]['starts']
            runs = save_dict['instances'][k]['runs']

            save_dict['instances'][k]['rle'] = \
            rle_to_string(starts, runs)

            del save_dict['instances'][k]['starts']
            del save_dict['instances'][k]['runs']

        for k,v in list(save_dict['instances'].items()):
            save_dict['instances'][str(k)] = v
            del save_dict['instances'][k]

        with open(savepath, mode='w') as handle:
            json.dump(save_dict, handle, indent=6)

    def load_from_json(self, fpath):
        with open(fpath, mode='r') as handle:
            load_dict = json.load(handle)

        # convert instance string coords to arrays
        for k in load_dict['instances'].keys():
            rle_string = load_dict['instances'][k]['rle']
            starts, runs = string_to_rle(rle_string)
            load_dict['instances'][k]['starts'] = starts
            load_dict['instances'][k]['runs'] = runs

        self.__dict__ = load_dict
