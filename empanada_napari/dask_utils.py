from torch.utils.data import Dataset

__all__ = [
    'DaskData',
]

def dask_take3d(array, indices, axis=0):
    """Similar to np.take but for 3d zarr arrays."""
    assert len(array.shape) == 3
    assert axis in [0, 1, 2]
    
    if axis == 0:
        x = array[indices]
    elif axis == 1:
        x = array[:, indices]
    else:
        x = array[..., indices]

    return x.compute()

class DaskData(Dataset):
    def __init__(self, array, axis=0, tfs=None):
        super(DaskData, self).__init__()
        self.array = array
        self.axis = axis
        self.tfs = tfs
        
    def __len__(self):
        return self.array.shape[self.axis]
    
    def __getitem__(self, idx):
        # load the image
        image = dask_take3d(self.array, idx, self.axis)
        image = self.tfs(image=image)['image']
        
        return {'index': idx, 'image': image}