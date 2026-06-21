"""Fully deferred variable for fast dataset opening.

By default a backend that supports it (currently h5netcdf) hands xarray a
dataset whose variables are :py:class:`LazyVariable` placeholders. Opening then
reads essentially nothing per variable -- only the variable names. Each
variable's dimensions, dtype, attributes, CF decoding and data are read from the
store (and cached) the first time that variable is touched. Set
``XARRAY_DISABLE_LAZY_OPEN`` to fall back to the eager path.

This trades eager validation for speed: inconsistencies that an eager open would
have raised immediately (e.g. a dimension used with conflicting sizes) instead
surface when the offending variable is first materialized. The dataset's
``sizes`` are accumulated as a running tally as variables are materialized.

Robustness note: xarray reads ``Variable._data`` / ``._dims`` / ``._attrs`` /
``._encoding`` (the ``__slots__``) directly in many places. ``LazyVariable``
therefore shadows those four slots with properties that materialize on first
access, so *any* access path -- not just the public ``.data`` / ``.dims`` -- is
covered.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any

from xarray.core import indexing
from xarray.core.variable import Variable


class _GrowingDims(dict):
    """A dataset's dimension-size mapping that completes itself on demand.

    Subclasses ``dict`` (so it satisfies xarray's ``isinstance(_dims, dict)``
    expectations) but starts empty. A dimension's size is filled in when a
    variable using it materializes. Any access that needs the *complete* set of
    dimensions -- iterating, ``len``, ``keys``/``values``/``items``, or looking
    up a dimension not seen yet -- triggers a one-time full materialization of
    every variable (via the registered finalizer), after which it behaves like
    an ordinary complete dict. Looking up a dimension already present stays
    cheap and does not force completion.

    This lets the common case (open, read a few variables) stay lazy while
    questions that genuinely need every variable (``ds.sizes``, alignment,
    indexing by an as-yet-unread dimension, ``assert_identical``) pay the full
    cost once, for all future calls.
    """

    def __init__(self) -> None:
        super().__init__()
        self._finalizer: Callable[[], None] | None = None

    def set_finalizer(self, finalizer: Callable[[], None]) -> None:
        self._finalizer = finalizer

    def _finalize(self) -> None:
        # a None finalizer means we are still being built: do nothing (and do
        # not latch), so a later access completes correctly
        finalizer = self._finalizer
        if finalizer is not None:
            self._finalizer = None
            finalizer()

    def __missing__(self, key):
        self._finalize()
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if dict.__contains__(self, key):
            return True
        self._finalize()
        return dict.__contains__(self, key)

    def __iter__(self) -> Iterator[Hashable]:
        self._finalize()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._finalize()
        return dict.__len__(self)

    def keys(self):
        self._finalize()
        return dict.keys(self)

    def values(self):
        self._finalize()
        return dict.values(self)

    def items(self):
        self._finalize()
        return dict.items(self)

    def copy(self) -> dict:
        self._finalize()
        return dict(self)


class _LazyVariableLoader:
    """Picklable ``() -> Variable`` reading and CF-decoding one variable.

    A plain closure would make lazy datasets unpicklable (breaking dask /
    multiprocessing); this object pickles via the (picklable) store.
    """

    __slots__ = ("decode_kwargs", "name", "store")

    def __init__(self, store, name: Hashable, decode_kwargs: dict) -> None:
        self.store = store
        self.name = name
        self.decode_kwargs = decode_kwargs

    def __call__(self) -> Variable:
        from xarray.conventions import decode_cf_variable

        var = self.store.open_store_variable_by_name(self.name)
        decode_coords = self.decode_kwargs.get("decode_coords", True)
        var = decode_cf_variable(
            self.name,
            var,
            mask_and_scale=self.decode_kwargs["mask_and_scale"],
            decode_times=self.decode_kwargs["decode_times"],
            concat_characters=self.decode_kwargs["concat_characters"],
            use_cftime=self.decode_kwargs["use_cftime"],
            decode_timedelta=self.decode_kwargs["decode_timedelta"],
        )
        if decode_coords in [True, "coordinates", "all"]:
            # mirror conventions.decode_cf_variables: move coordinates attr to
            # encoding so it is not treated as a regular attribute
            if "coordinates" in var.attrs:
                var.encoding["coordinates"] = var.attrs.pop("coordinates")
        return var


class LazyVariable(Variable):
    """A :py:class:`~xarray.Variable` materialized from a store on first access.

    Build lazy instances with :py:meth:`_make_lazy`. The normal constructor is
    kept signature-compatible with :py:class:`~xarray.Variable` (producing an
    already-materialized instance) so the many internal ``type(var)(dims, ...)``
    call sites keep working.
    """

    __slots__ = (
        "_attrs_real",
        "_data_real",
        "_dim_sizes",
        "_dims_real",
        "_encoding_real",
        "_loaded",
        "_loader",
        "_protect_cache",
    )

    def __init__(
        self, dims=None, data=None, attrs=None, encoding=None, fastpath=False
    ) -> None:
        # initialize the backing slots before super().__init__ writes them
        self._loader = None
        self._loaded = True
        self._dim_sizes = None
        self._protect_cache = None
        self._dims_real = None
        self._data_real = None
        self._attrs_real = None
        self._encoding_real = None
        super().__init__(dims, data, attrs, encoding, fastpath=fastpath)

    @classmethod
    def _make_lazy(
        cls,
        loader: Callable[[], Variable],
        dim_sizes: dict[Hashable, int],
    ) -> LazyVariable:
        """Create a deferred variable.

        Parameters
        ----------
        loader : callable
            ``() -> Variable`` returning the fully read and CF-decoded variable.
            Called at most once; the result is cached in place.
        dim_sizes : dict
            The owning dataset's ``_dims`` mapping, updated with this variable's
            dimension sizes when it materializes (the running dimension tally).
        """
        self = cls.__new__(cls)
        self._loader = loader
        self._loaded = False
        self._dim_sizes = dim_sizes
        # None => do not wrap data; True/False => wrap, with/without memory cache
        self._protect_cache = None
        self._dims_real = None
        self._data_real = None
        self._attrs_real = None
        self._encoding_real = None
        return self

    def _set_protect(self, cache: bool) -> None:
        """Record that this variable's data should be protected on materialize.

        Deferred from :func:`backends.api._protect_dataset_variables_inplace`,
        which cannot wrap the (not-yet-read) data of a lazy variable up front.
        """
        self._protect_cache = cache

    def _ensure_materialized(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        var = self._loader()
        # running dimension tally with a deferred consistency check
        dim_sizes = self._dim_sizes
        for dim, size in zip(var.dims, var.shape, strict=True):
            previous = dim_sizes.get(dim)
            if previous is not None and previous != size:
                raise ValueError(
                    f"conflicting sizes for dimension {dim!r}: "
                    f"size {previous} but variable {self!r} has size {size}"
                )
            dim_sizes[dim] = size
        self._dims_real = var._dims
        self._attrs_real = var._attrs
        self._encoding_real = var._encoding
        data = var._data
        if self._protect_cache is not None:
            data = indexing.CopyOnWriteArray(data)
            if self._protect_cache:
                data = indexing.MemoryCachedArray(data)
        self._data_real = data

    # Shadow Variable's __slots__ with properties that materialize on access, so
    # every direct slot read (not just the public .data/.dims/...) is covered.
    @property
    def _dims(self):
        self._ensure_materialized()
        return self._dims_real

    @_dims.setter
    def _dims(self, value) -> None:
        self._dims_real = value

    @property
    def _data(self):
        self._ensure_materialized()
        return self._data_real

    @_data.setter
    def _data(self, value) -> None:
        self._data_real = value

    @property
    def _attrs(self) -> dict[Any, Any] | None:
        self._ensure_materialized()
        return self._attrs_real

    @_attrs.setter
    def _attrs(self, value: Mapping[Any, Any] | None) -> None:
        self._attrs_real = value

    @property
    def _encoding(self) -> dict[Any, Any] | None:
        self._ensure_materialized()
        return self._encoding_real

    @_encoding.setter
    def _encoding(self, value: Mapping[Any, Any] | None) -> None:
        self._encoding_real = value
