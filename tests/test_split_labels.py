import numpy as np
from napari.components import ViewerModel

from empanada_napari._merge_split_widget import split_labels


def test_split_disconnected_pieces_of_the_same_label():
    """Two exclaves of one label must split instead of 'Nothing to split.'

    The label crop is a tight bounding box, so each piece sits on the crop
    border. peak_local_max used to drop those peaks.
    """
    labels = np.zeros((80, 80), dtype=np.int32)
    labels[10:30, 10:30] = 5
    labels[50:70, 50:70] = 5

    viewer = ViewerModel()
    labels_layer = viewer.add_labels(labels, name="labels")
    points_layer = viewer.add_points(np.array([[20, 20], [60, 60]]), name="points")

    widget = split_labels()
    widget(
        viewer=viewer,
        labels_layer=labels_layer,
        points_layer=points_layer,
        min_distance=10,
        points_as_markers=False,
        apply3d=False,
        new_label=False,
        start_label="",
    )

    result = labels_layer.data
    piece_a = result[20, 20]
    piece_b = result[60, 60]
    assert piece_a != 0 and piece_b != 0
    assert piece_a != piece_b
    assert piece_a != 5 and piece_b != 5
