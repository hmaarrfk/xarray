from collections import Mapping
from contextlib import contextmanager
import pandas as pd

from . import formatting
from .utils import Frozen
from .merge import merge_coords, merge_coords_without_align
from .pycompat import iteritems, basestring, OrderedDict
from .variable import default_index_coordinate


class AbstractCoordinates(Mapping, formatting.ReprMixin):
    def __getitem__(self, key):
        if (key in self._names or
            (isinstance(key, basestring) and
             key.split('.')[0] in self._names)):
            # allow indexing current coordinates or components
            return self._data[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        self.update({key: value})

    @property
    def indexes(self):
        return self._data.indexes

    @property
    def variables(self):
        raise NotImplementedError

    def _update_coords(self, coords):
        raise NotImplementedError

    def __iter__(self):
        # needs to be in the same order as the dataset variables
        for k in self.variables:
            if k in self._names:
                yield k

    def __len__(self):
        return len(self._names)

    def __contains__(self, key):
        return key in self._names

    def __unicode__(self):
        return formatting.coords_repr(self)

    @property
    def dims(self):
        return self._data.dims

    def to_index(self, ordered_dims=None):
        """Convert all index coordinates into a :py:class:`pandas.MultiIndex`
        """
        if ordered_dims is None:
            ordered_dims = self.dims
        indexes = [self.variables[k].to_index() for k in ordered_dims]
        return pd.MultiIndex.from_product(indexes, names=list(ordered_dims))

    def update(self, other):
        other_vars = getattr(other, 'variables', other)
        coords = merge_coords([self.variables, other_vars],
                              priority_arg=1, indexes=self.indexes)
        self._update_coords(coords)

    def _merge_raw(self, other):
        """For use with binary arithmetic."""
        if other is None:
            variables = OrderedDict(self.variables)
        else:
            # don't align because we already called xarray.align
            variables = merge_coords_without_align(
                [self.variables, other.variables])
        return variables

    @contextmanager
    def _merge_inplace(self, other):
        """For use with in-place binary arithmetic."""
        if other is None:
            yield
        else:
            # don't include indexes in priority_vars, because we didn't align
            # first
            priority_vars = OrderedDict(
                (k, v) for k, v in self.variables.items() if k not in self.dims)
            variables = merge_coords_without_align(
                [self.variables, other.variables], priority_vars=priority_vars)
            yield
            self._update_coords(variables)

    def merge(self, other):
        """Merge two sets of coordinates to create a new Dataset

        The method implements the logic used for joining coordinates in the
        result of a binary operation performed on xarray objects:

        - If two index coordinates conflict (are not equal), an exception is
          raised. You must align your data before passing it to this method.
        - If an index coordinate and a non-index coordinate conflict, the non-
          index coordinate is dropped.
        - If two non-index coordinates conflict, both are dropped.

        Parameters
        ----------
        other : DatasetCoordinates or DataArrayCoordinates
            The coordinates from another dataset or data array.

        Returns
        -------
        merged : Dataset
            A new Dataset with merged coordinates.
        """
        from .dataset import Dataset

        if other is None:
            return self.to_dataset()
        else:
            other_vars = getattr(other, 'variables', other)
            coords = merge_coords_without_align([self.variables, other_vars])
            return Dataset._from_vars_and_coord_names(coords, set(coords))


class DatasetCoordinates(AbstractCoordinates):
    """Dictionary like container for Dataset coordinates.

    Essentially an immutable OrderedDict with keys given by the array's
    dimensions and the values given by the corresponding xarray.Coordinate
    objects.
    """
    def __init__(self, dataset):
        self._data = dataset

    @property
    def _names(self):
        return self._data._coord_names

    @property
    def variables(self):
        return Frozen(OrderedDict((k, v)
                                  for k, v in self._data.variables.items()
                                  if k in self._names))

    def to_dataset(self):
        """Convert these coordinates into a new Dataset
        """
        return self._data._copy_listed(self._names)

    def _update_coords(self, coords):
        from .dataset import calculate_dimensions

        variables = self._data._variables.copy()
        variables.update(coords)

        # check for inconsistent state *before* modifying anything in-place
        dims = calculate_dimensions(variables)
        for dim, size in dims.items():
            if dim not in variables:
                variables[dim] = default_index_coordinate(dim, size)

        updated_coord_names = set(coords) | set(dims)

        self._data._variables = variables
        self._data._coord_names.update(updated_coord_names)
        self._data._dims = dict(dims)

    def __delitem__(self, key):
        if key in self:
            del self._data[key]
        else:
            raise KeyError(key)


class DataArrayCoordinates(AbstractCoordinates):
    """Dictionary like container for DataArray coordinates.

    Essentially an OrderedDict with keys given by the array's
    dimensions and the values given by corresponding DataArray objects.
    """
    def __init__(self, dataarray):
        self._data = dataarray

    @property
    def _names(self):
        return set(self._data._coords)

    def _update_coords(self, coords):
        from .dataset import calculate_dimensions

        dims = calculate_dimensions(coords)
        if set(dims) != set(self.dims):
            raise ValueError('cannot add coordinates with new dimensions to '
                             'a DataArray')
        self._data._coords = coords

    @property
    def variables(self):
        return Frozen(self._data._coords)

    def _to_dataset(self, shallow_copy=True):
        from .dataset import Dataset
        coords = OrderedDict((k, v.copy(deep=False) if shallow_copy else v)
                             for k, v in self._data._coords.items())
        return Dataset._from_vars_and_coord_names(coords, set(coords))

    def to_dataset(self):
        return self._to_dataset()

    def __delitem__(self, key):
        if key in self.dims:
            raise ValueError('cannot delete a coordinate corresponding to a '
                             'DataArray dimension')
        del self._data._coords[key]


class DataArrayLevelCoordinates(AbstractCoordinates):
    """Dictionary like container for DataArray MultiIndex level coordinates.

    Used for attribute style lookup. Not returned directly by any
    public methods.
    """
    def __init__(self, dataarray):
        self._data = dataarray

    @property
    def _names(self):
        return set(self._data._level_coords)

    @property
    def variables(self):
        level_coords = OrderedDict(
            (k, self._data[v].variable.get_level_variable(k))
             for k, v in self._data._level_coords.items())
        return Frozen(level_coords)


class Indexes(Mapping, formatting.ReprMixin):
    """Ordered Mapping[str, pandas.Index] for xarray objects.
    """
    def __init__(self, variables, dims):
        """Not for public consumption.

        Arguments
        ---------
        variables : OrderedDict
            Reference to OrderedDict holding variable objects. Should be the
            same dictionary used by the source object.
        dims : sequence or mapping
            Should be the same dimensions used by the source object.
        """
        self._variables = variables
        self._dims = dims

    def __iter__(self):
        return iter(self._dims)

    def __len__(self):
        return len(self._dims)

    def __contains__(self, key):
        return key in self._dims

    def __getitem__(self, key):
        if key in self:
            return self._variables[key].to_index()
        else:
            raise KeyError(key)

    def __unicode__(self):
        return formatting.indexes_repr(self)
