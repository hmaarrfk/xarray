from __future__ import annotations

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING

from xarray import conventions
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendEntrypoint,
    T_PathFileOrDataStore,
)
from xarray.core.coordinates import Coordinates
from xarray.core.dataset import Dataset

if TYPE_CHECKING:
    pass


def _lazy_open_enabled() -> bool:
    # Default on: defer all per-variable reads (dimensions, attributes, CF
    # decoding and data) until each variable is first accessed. Minimizes open
    # time at the cost of eager validation -- see backends.lazy.LazyVariable.
    # Set XARRAY_DISABLE_LAZY_OPEN to fall back to the eager path.
    return not os.environ.get("XARRAY_DISABLE_LAZY_OPEN")


class StoreBackendEntrypoint(BackendEntrypoint):
    description = "Open AbstractDataStore instances in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.StoreBackendEntrypoint.html"

    def guess_can_open(self, filename_or_obj: T_PathFileOrDataStore) -> bool:
        return isinstance(filename_or_obj, AbstractDataStore)

    def open_dataset(
        self,
        filename_or_obj: T_PathFileOrDataStore,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        set_indexes: bool = True,
        use_cftime=None,
        decode_timedelta=None,
    ) -> Dataset:
        assert isinstance(filename_or_obj, AbstractDataStore)

        if (
            getattr(filename_or_obj, "supports_lazy_load", False)
            and _lazy_open_enabled()
            # ``decode_coords="all"`` also promotes grid_mapping / bounds /
            # cell_measures targets; defer to the eager path for that case.
            and decode_coords != "all"
        ):
            return self._open_dataset_lazy(
                filename_or_obj,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )

        vars, attrs = filename_or_obj.load()
        encoding = filename_or_obj.get_encoding()

        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars,
            attrs,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        # split data and coordinate variables (promote dimension coordinates)
        data_vars = {}
        coord_vars = {}
        for name, var in vars.items():
            if name in coord_names or var.dims == (name,):
                coord_vars[name] = var
            else:
                data_vars[name] = var

        # explicit Coordinates object with no index passed
        coords = Coordinates(coord_vars, indexes={})

        ds = Dataset(data_vars, coords=coords, attrs=attrs)
        ds.set_close(filename_or_obj.close)
        ds.encoding = encoding

        return ds

    def _open_dataset_lazy(
        self,
        store: AbstractDataStore,
        *,
        mask_and_scale,
        decode_times,
        concat_characters,
        decode_coords,
        drop_variables,
        use_cftime,
        decode_timedelta,
    ) -> Dataset:
        """Open a dataset deferring data-variable reads until first access.

        Data variables are read (and cached) the first time they are touched --
        only their names are read up front. Coordinates and their indexes are
        built eagerly, so the dataset is fully usable (``.sel`` / ``.xindexes``)
        and matches an eagerly opened one. ``ds.sizes`` is seeded from the
        file's dimensions and kept as a running tally as variables materialize.
        See ``backends.lazy``.
        """
        from xarray.backends.common import _decode_variable_name
        from xarray.backends.lazy import (
            LazyVariable,
            _GrowingDims,
            _LazyVariableLoader,
        )

        if isinstance(drop_variables, str):
            drop_variables = [drop_variables]
        elif drop_variables is None:
            drop_variables = []
        drop_variables = set(drop_variables)

        # map each (decoded) variable name to the raw name used to read it
        raw_names = {
            _decode_variable_name(n): n
            for n in store.get_variable_names()
            if n not in drop_variables
        }
        names = list(raw_names)
        name_set = set(names)
        attrs = dict(store.get_attrs())
        encoding = store.get_encoding()
        dim_names = set(store.get_dimension_names())

        # ``_dims`` doubles as the running dimension-size tally: it accumulates a
        # dimension's size when a variable using it materializes, and completes
        # itself (materializing every variable, once) when the full set of
        # dimensions is actually needed. Seeding it with every file dimension up
        # front instead would include dimensions used by no surviving variable
        # (e.g. helper dimensions consumed by character-array decoding).
        dim_sizes = _GrowingDims()

        decode_kwargs = {
            "mask_and_scale": mask_and_scale,
            "decode_times": decode_times,
            "concat_characters": concat_characters,
            "decode_coords": decode_coords,
            "use_cftime": use_cftime,
            "decode_timedelta": decode_timedelta,
        }

        variables: dict = {
            n: LazyVariable._make_lazy(
                _LazyVariableLoader(store, raw_names[n], decode_kwargs), dim_sizes
            )
            for n in names
        }

        # Auxiliary coordinates declared via a "coordinates" attribute, on
        # individual variables or globally on the dataset (orphan coordinates).
        aux_coord_names: set = set()
        if decode_coords in [True, "coordinates"]:
            aux_coord_names = {c for c in store.get_coordinate_names() if c in name_set}
            global_coords = attrs.get("coordinates")
            if isinstance(global_coords, str):
                aux_coord_names |= {c for c in global_coords.split() if c in name_set}
                del attrs["coordinates"]

        # Coordinate variables are those named like a dimension (xarray promotes
        # any such variable to a coordinate, even multidimensional ones) and the
        # auxiliary coordinates. xarray requires dimension coordinates to be
        # eagerly materialized IndexVariables, so coordinates are read here while
        # data variables stay lazy.
        coord_vars: dict = {
            name: variables[name]
            for name in names
            if name in dim_names or name in aux_coord_names
        }

        # Build the Coordinates object without indexes; default indexes are
        # created afterwards by backends.api._maybe_create_default_indexes, which
        # honors the ``create_default_indexes`` argument.
        coords = Coordinates(coord_vars, indexes={})
        variables.update(coords.variables)
        coord_names = set(coords.variables)

        # completing ``dim_sizes`` means materializing every remaining variable
        def _materialize_all_dims():
            for var in variables.values():
                if isinstance(var, LazyVariable):
                    var._ensure_materialized()

        dim_sizes.set_finalizer(_materialize_all_dims)

        ds = Dataset._construct_direct(
            variables,
            coord_names,
            dims=dim_sizes,
            attrs=attrs,
            indexes={},
        )
        ds.set_close(store.close)
        ds.encoding = encoding
        return ds


BACKEND_ENTRYPOINTS["store"] = (None, StoreBackendEntrypoint)
