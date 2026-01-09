import pytest
import numpy as np
from empanada.array_utils import _box_iou, intersection_from_ranges, split_range_by_votes, \
                                extend_range, rle_voting, _join_ranges, invert_ranges


@pytest.mark.parametrize(['box1', 'box2', 'expected'],
    [([[0, 0, 20, 20]], [[5, 5, 25, 25]], [[0], [0], [0.39], [225]]),
     ([[0, 0, 20, 20]], [[30, 0, 50, 20]], [[], [], [], []])],
     ids=["overlapping_boxes", "non-overlapping_boxes"])
def test__box_iou(box1, box2, expected):
    """
    Test `_box_iou` function.

    The function should:
    - read in two arrays of box coordinates
    - compute intersection over union (IoU)
    - return a list of: intersecting rows, intersecting columns, \
      IoU, and intersection area

    Expected:
    - A list containing 4 sub-lists
    """
            
    box_iou = _box_iou(np.array(box1), np.array(box2))
    for i in range(len(box_iou)):
        assert box_iou[i] == pytest.approx(expected[i], abs=0.02)


@pytest.mark.parametrize(['merged_runs', 'changes', 'expected'],
    [([[0, 10], [7, 20]], [True], 3),
     ([[0, 10], [7, 20]], [False], 0)],
     ids=["different_source", "same_source"])
def test_intersection_from_ranges(merged_runs, changes, expected):
    """
    Test `intersection_from_ranges` function.

    The function should:
    - read in an array of merged runs, where each element is a range of [start, end]
    - read in a boolean array of changes
    - compute number of overlapping pixels/voxels in merged runs array
    - return an integer number of overlapping pixels/voxels

    Expected:
    - An integer value
    """
    intersections = intersection_from_ranges(np.array(merged_runs), np.array(changes))
    assert intersections == expected


@pytest.mark.parametrize(['running_range', 'num_votes', 'expected'],
    [([0, 10], [2,3,3,3,1,2,2,3,3,4], [[0, 4], [5, 10]]),
     ([0, 10], [2,3,3,3,2,2,2,3,3,4], [[0, 10]])],
     ids=["vote_under_threshold", "vote_over_threshold"])
def test_split_range_by_votes(running_range, num_votes, expected):
    """
    Test `split_range_by_votes` function.

    The function should:
    - read in a running range of [start, end], and a list of len(end-start) containing votes
    - split the running range if the vote at a given index in the range is below a threshold
    - return a list of ranges containing [start, end]

    Expected:
    - A list of [start, end] lists
    """
    ranges = split_range_by_votes(np.array(running_range), np.array(num_votes), vote_thr=2)
    ranges = [list(r) for r in ranges]
    assert ranges == expected

@pytest.mark.parametrize(['range1', 'range2', 'num_votes', 'expected'],
    [([1, 10], [3, 10], [2,4,4,4,4,2,2,2,2,2], ([1, 10], [2, 4, 5, 5, 5, 3, 3, 3, 3, 3]))],
    ids=["overlapping_ranges"])
def test_extend_range(range1, range2, num_votes, expected):
    """
    Test `extend_range` function.

    The function should:
    - read in two ranges of [start, end], and a list of len(end-start) containing votes
    - merge the two ranges into one
    - compute a list of len(new end-new start) containing votes covering the new range
    - return a range list containing [start, end], and an integer list of votes

    Expected:
    - A list of [start, end] and an integer list of len(end-start)
    """
    range1 = np.array(range1)
    range2 = np.array(range2)
    num_votes = np.array(num_votes)

    ranges = extend_range(range1, range2, num_votes)
    ranges = tuple(list(r) for r in ranges)
    assert ranges == expected


@pytest.mark.parametrize(['ranges', 'expected'],
    [([(10,20), (7, 26)], [[10, 20], [23, 26]])],
    ids=["overlapping_ranges"])
def test_rle_voting(ranges, expected):
    """
    Test `rle_voting` function.

    The function should:
    - read in a list of ranges of (start, end)
    - compute the number of votes at each index within overlapping ranges
    - return a list containing [start, end] ranges where all indices votes \
      were greater than or equal to the vote threshold

    Expected:
    - A list of [start, end] ranges
    """
    new_ranges = rle_voting(np.array(ranges))
    new_ranges = [list(r) for r in new_ranges]
    assert new_ranges == expected


@pytest.mark.parametrize(['ranges', 'expected'],
    [([(0,10), (6, 10)], [[0, 10]]),
     ([(0,10), (11, 20)], [[0, 10], [11, 20]]),
     ([(0,10), (10, 20)], [[0, 20]])],
    ids=["overlapping_ranges", "non_overlapping_ranges",
         "border_ranges"])
def test__join_ranges(ranges, expected):
    """
    Test `_join_ranges` function.

    The function should:
    - read in a list of ranges of (start, end)
    - return an array containing a joined [start, end] range

    Expected:
    - A list of [start, end] ranges
    """
    new_ranges = _join_ranges(np.array(ranges))
    assert np.array_equal(new_ranges, expected)


@pytest.mark.parametrize(['ranges', 'size', 'expected'],
    [([(2, 6), (4, 12)], 15, [[0, 2], [6, 4], [12, 15]])],
    ids=["overlapping_ranges"])
def test_invert_ranges(ranges, size, expected):
    """
    Test `invert_ranges` function.

    The function should:
    - read in a list of ranges of (start, end)
    - read in an integer size
    - compute inverted ranges where size is the last range's end index
    - return an array containing reversed [start, end] ranges

    Expected:
    - A list of [start, end] ranges
    """
    new_ranges = invert_ranges(np.array(ranges), size)
    assert np.array_equal(new_ranges, expected)