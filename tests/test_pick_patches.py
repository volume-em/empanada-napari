import numpy as np
from napari.layers import Points

from empanada_napari._pick_patches import pick_patches


def test_point_count_tracks_all_slices_and_active_layer():
    first_layer = Points(
        np.array(
            [
                [0, 10, 10],
                [1, 20, 20],
                [2, 30, 30],
            ]
        ),
        name='first',
    )
    second_layer = Points(np.array([[5, 40, 40]]), name='second')

    widget = pick_patches()
    assert widget.point_count.label == 'Points (all slices)'
    assert 'not currently visible' in widget.point_count.tooltip

    widget.points_layer.choices = [first_layer, second_layer]
    widget.points_layer.value = first_layer

    assert widget.point_count.value == '3'

    first_layer.add([3, 40, 40])
    assert widget.point_count.value == '4'

    first_layer.data = first_layer.data[:-1]
    assert widget.point_count.value == '3'

    widget.points_layer.value = second_layer
    assert widget.point_count.value == '1'

    first_layer.add([4, 50, 50])
    assert widget.point_count.value == '1'

    second_layer.add([6, 50, 50])
    assert widget.point_count.value == '2'
