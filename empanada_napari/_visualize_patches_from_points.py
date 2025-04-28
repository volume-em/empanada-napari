import napari
from magicgui import magicgui



def create_shapes_layer_from_points():

    gui_params = dict(
        tile_width=dict(widget_type='SpinBox', label='Tile width', min=1, max=10000, step=1, value=512, tooltip='Width of the tile in pixels'),
    )
    @magicgui(
        call_button='Visualize patches',
        layout='vertical',
        **gui_params
    )
    def widget(
            viewer: napari.viewer.Viewer,
            points_layer: napari.layers.Points,
            tile_width: int,
            shapes_layer: napari.layers.Shapes = None,  # Optional shapes layer to visualize patches (if provided
        ):

        """
        Visualize patches from points in a napari viewer.
        Parameters
        ----------
        viewer : napari.viewer.Viewer
            The napari viewer instance.
        points_layer : napari.layers.Points
            The points layer containing the points from which to create patches.
        tile_width : int
            The width of the patches to create (in pixels).
        output_to_layer : bool
            Whether to output the patches to a new napari layer.
        shapes_layer : napari.layers.Shapes, optional
            An optional shapes layer to visualize the patches. If provided, the patches will be added to this layer.
        """

        if points_layer is None:
            print("No points layer provided.")
            return

        # create shapes from points

        points = points_layer.data
        rectangles = []
        for point in points:
            if len(point.shape) != 2:
                x, y = point[-2], point[-1]
            else:
                x, y = point[0], point[1]

            # Get all four points coords
            x1 = int(x - tile_width / 2)
            y1 = int(y - tile_width / 2)
            x2 = int(x + tile_width / 2)
            y2 = int(y + tile_width / 2)
            # Create a rectangle shape from the points
            rectangle = [
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ]

            # add spatial location from the points layer
            for idx in range(4):
                rectangle[idx] = list(point)[:-2] + rectangle[idx]

            # Add the rectangle shape to the shapes layer
            rectangles.append(rectangle)

        if shapes_layer is None:
            # Create a new shapes layer if not provided or if output_to_layer is False
            shapes_layer = viewer.add_shapes(name='Patches', data=rectangles)
        else:
            shapes_layer.data = rectangles

        shapes_layer.visible = False
        shapes_layer.visible = True

    return widget




