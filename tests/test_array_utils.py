import numba
import pytest
import numpy as np
from pytest import param
from empanada.array_utils import _box_iou, intersection_from_ranges, split_range_by_votes, \
                                extend_range, rle_voting


@pytest.mark.parametrize(['box1', 'box2', 'expected'],
    [([[0, 0, 20, 20]], [[5, 5, 25, 25]], [[0], [0], [0.39], [225]]),
     ([[0, 0, 20, 20]], [[30, 0, 50, 20]], [[], [], [], []])])
def test__box_iou(box1, box2, expected) -> None:
    box_iou = _box_iou(np.array(box1), np.array(box2))

    for i in range(len(box_iou)):
        assert box_iou[i] == pytest.approx(expected[i], abs=0.02)


@pytest.mark.parametrize(['merged_runs', 'changes', 'expected'],
    [([[0, 10], [7, 20]], [True], 3),
     ([[0, 10], [7, 20]], [False], 0)])
def test_intersection_from_ranges(merged_runs, changes, expected):
    intersections = intersection_from_ranges(np.array(merged_runs), np.array(changes))
    assert intersections == expected


@pytest.mark.parametrize(['running_range', 'num_votes', 'expected'],
    [([0, 10], [2,3,3,3,1,2,2,3,3,4], [[0, 4], [5, 10]]),
     ([0, 10], [2,3,3,3,2,2,2,3,3,4], [[0, 10]])])
def test_split_range_by_votes(running_range, num_votes, expected):
    ranges = split_range_by_votes(np.array(running_range), np.array(num_votes), vote_thr=2)
    ranges = [list(r) for r in ranges]
    assert ranges == expected


@pytest.mark.parametrize(['range1', 'range2', 'num_votes', 'expected'],
    [([1, 10], [3, 10], [2,4,4,4,4,2,2,2,2,2], ([1, 10], [2, 4, 5, 5, 5, 3, 3, 3, 3, 3]))])
def test_extend_range(range1, range2, num_votes, expected):
    range1 = np.array(range1)
    range2 = np.array(range2)
    num_votes = np.array(num_votes)

    ranges = extend_range(range1, range2, num_votes)
    ranges = tuple(list(r) for r in ranges)
    assert ranges == expected


@pytest.mark.parametrize(['ranges', 'expected'],
    [([(10,20), (7, 26)], [[10, 20], [23, 26]])])
def test_rle_voting(ranges, expected):
    new_ranges = rle_voting(np.array(ranges))
    new_ranges = [list(r) for r in new_ranges]
    assert new_ranges == expected
