"""Provides a nested dictionary that stores hierarchical mappings.

.. autosummary::
   :nosignatures:

   NestedDict

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import Any, Generic, Literal, TypeAlias, TypeVar, Union, overload

from typing_extensions import Self

# values are of generic type TValue, which will be specified
TValue = TypeVar("TValue")
# trees are nested dicts
TNestedDict: TypeAlias = "NestedDict[TValue]"
# nodes are trees or values
TNestedDictValue = Union[TValue, TNestedDict]  # noqa: UP007
# the dictionary version of the entire tree can have subtrees
TDictTree = dict[str, Union[TValue, "TDictTree"]]

T = TypeVar


class NestedDict(MutableMapping[str, TNestedDictValue], Generic[TValue]):
    """Stores hierarchical mappings with string paths as keys.

    `NestedDict` wraps nested mappings and supports reading and writing nested
    values using a separator-based key syntax (for example ``"a.b.c"``). It can
    convert between flat and nested representations and recursively traverses
    children when requested.

    Note:
        Equivalent entries can overwrite each other during initialization.
        For instance, ``NestedDict({'a.b': 1, 'a': {'b': 2}})`` stores only one
        final value for ``a.b``.
    """

    sep: str = "."
    """str: Separator used in key paths to traverse nested levels."""
    data: MutableMapping[str, TNestedDictValue]
    """dict: Internal mapping storing top-level keys and values for this instance."""

    def __init__(
        self, data: MutableMapping[str, TNestedDictValue] | None = None
    ) -> None:
        """Initializes a nested dictionary from an optional mapping.

        Args:
            data (MutableMapping[str, Any] | None):
                Optional mapping used to populate the instance. Nested plain
                dictionaries are converted into `NestedDict` children.
        """
        self.data = self._make_dict()
        if data is not None:
            self.update_recursive(data)

    def _make_dict(self) -> MutableMapping[str, TNestedDictValue]:
        """Create the backing mapping used to store top-level entries."""
        return {}

    def _make_node(self) -> Self:
        """Create an empty child node of the current mapping type."""
        return self.__class__()

    def _node(self, key: str, *, parent: str = "") -> tuple[TNestedDict, str, bool]:
        """Resolve a key path to the owning node and local key.

        Args:
            key:
                Key or nested key path to resolve.
            parent:
                Prefix accumulated while recursing through nested nodes.

        Returns:
            tuple[NestedDict[TValue], str, bool]:
                The owning node, the final local key, and whether the resolved value
                is itself a nested node.
        """
        if not isinstance(key, str):
            msg = f"Keys must be strings, not {key!r}"
            raise TypeError(msg)
        if self.sep not in key:
            # key denotes a node
            try:
                node = self.data[key]
            except KeyError as err:
                # node did not exist
                msg = f"`{parent}{key}` not in {list(self.data.keys())}"
                raise KeyError(msg) from err
            is_tree = isinstance(node, NestedDict)
            return self, key, is_tree

        # key denotes entire branch
        child, grandchildren = key.split(self.sep, 1)
        try:
            node = self.data[child]  # next node in branch
        except KeyError as err:
            # node did not exist
            msg = f"`{parent}{key}` not in {list(self.data.keys())}"
            raise KeyError(msg) from err
        if not isinstance(node, NestedDict):
            msg = f"`{child}` is not a tree node."
            raise TypeError(msg)
        # traverse branch recursively
        return node._node(grandchildren, parent=parent + key + self.sep)

    def __getitem__(self, key: str) -> TNestedDictValue:
        """Returns an item using dictionary indexing syntax.

        Args:
            key (str):
                Key or nested key path to resolve.

        Returns:
            Any:
                Value associated with `key`.
        """
        node, subkey, _ = self._node(key)
        res = node.data[subkey]
        return res

    def __setitem__(self, key: str, value: TNestedDictValue) -> None:
        """Assigns a value to a key or nested key path.

        Args:
            key (str):
                Target key. If it contains the separator, missing intermediate
                `NestedDict` nodes are created automatically.
            value (Any):
                Value to store.

        Raises:
            TypeError:
                If path assignment traverses a non-`NestedDict` child.
        """
        # prepare keys and values
        if not isinstance(key, str):
            msg = "Keys must be strings"
            raise TypeError(msg)

        try:
            node, subkey, is_tree = self._node(key)
        except KeyError:
            # entry does not exist yet
            if self.sep in key:
                # create parents
                node_key, value_key = key.rsplit(self.sep, 1)
                subnode: TNestedDict = self.create_node(node_key)
            else:
                subnode, value_key = self, key
            subnode.data[value_key] = value
        else:
            # update existing entry
            if is_tree:
                node.data[subkey].update_recursive(value)
            elif isinstance(value, MutableMapping):
                msg = "Cannot replace normal value with tree"
                raise TypeError(msg)
            else:
                node.data[subkey] = value

    def __delitem__(self, key: str) -> None:
        """Deletes an item addressed by a simple or nested key.

        Args:
            key (str):
                Key or key path identifying the value to remove.

        Raises:
            KeyError:
                If the key path cannot be resolved.
        """
        node, subkey, _ = self._node(key)
        del node.data[subkey]

    def __contains__(self, key) -> bool:
        """Checks whether a key or key path is present.

        Args:
            key (object):
                Candidate key to test. The implementation accepts only strings.

        Returns:
            bool:
                `True` if the key path exists, otherwise `False`.
        """
        if not isinstance(key, str):
            return False
        node = self
        for node_key in key.split(self.sep):
            try:
                node = node[node_key]
            except (KeyError, TypeError):
                return False
        return True

    def __len__(self) -> int:
        """Returns the number of top-level keys.

        Returns:
            int:
                Number of entries stored at the current level.
        """
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        """Iterates over top-level keys.

        Returns:
            Iterator[str]:
                Iterator yielding top-level keys.
        """
        return self.data.__iter__()

    def clear(self) -> None:
        """Removes all top-level entries from the mapping."""
        self.data.clear()

    @overload  # type: ignore
    def values(
        self, *, flatten: Literal[False] = False
    ) -> Iterator[TNestedDictValue]: ...
    @overload
    def values(self, *, flatten: Literal[True]) -> Iterator[TValue]: ...

    def values(self, *, flatten: bool = False) -> Iterator[TNestedDictValue]:
        """Iterates over values, optionally recursing into nested children.

        Args:
            flatten (bool):
                If `True`, yields values from all descendant `NestedDict`
                instances. If `False`, yields only top-level values.

        Returns:
            Iterator[Any]:
                Iterator over values according to `flatten`.
        """
        if flatten:
            for value in self.data.values():
                if isinstance(value, NestedDict):
                    yield from value.values(flatten=True)  # recurse into sub dictionary
                else:
                    yield value
        else:
            yield from self.data.values()

    def keys(self, *, flatten: bool = False) -> Iterator[str]:  # type: ignore
        """Iterates over keys, optionally returning flattened key paths.

        Args:
            flatten (bool):
                If `True`, yields separator-joined paths for descendant keys.
                If `False`, yields only top-level keys.

        Returns:
            Iterator[str]:
                Iterator over keys or flattened key paths.

        Raises:
            TypeError:
                If a key used during flattening is not a string.
        """
        if flatten:
            for key, value in self.data.items():
                if isinstance(value, NestedDict):
                    # recurse into sub dictionary
                    for k in value.keys(flatten=True):
                        yield key + self.sep + k
                else:
                    yield key
        else:
            yield from self.data.keys()

    @overload  # type: ignore
    def items(
        self, *, flatten: Literal[False] = False
    ) -> Iterator[tuple[str, TNestedDictValue]]: ...
    @overload
    def items(self, *, flatten: Literal[True]) -> Iterator[tuple[str, TValue]]: ...

    def items(self, *, flatten: bool = False) -> Iterator[tuple[str, TNestedDictValue]]:
        """Iterates over key-value pairs, optionally flattening nested paths.

        Args:
            flatten (bool):
                If `True`, yields `(path, value)` pairs for all descendants.
                If `False`, yields only top-level pairs.

        Returns:
            Iterator[tuple[str, Any]]:
                Iterator over key-value pairs according to `flatten`.

        Raises:
            TypeError:
                If a key used during flattening is not a string.
        """
        if flatten:
            for key, value in self.data.items():
                if isinstance(value, NestedDict):
                    # recurse into sub dictionary
                    for k, v in value.items(flatten=True):
                        yield key + self.sep + k, v
                else:
                    yield key, value
        else:
            yield from self.data.items()

    def __repr__(self) -> str:
        """Builds a debug representation of this mapping.

        Returns:
            str:
                String containing the class name and internal data mapping.
        """
        return f"{self.__class__.__name__}({self.data!r})"

    def create_node(self, key: str) -> Self:
        """Create an empty node at the given location.

        Creates all necessary parent nodes recursively. Skips nodes that already exist.

        Args:
            key:
                Key or nested key path identifying the node to create.

        Returns:
            The leaf node
        """
        if not isinstance(key, str):
            msg = "Keys must be strings"
            raise TypeError(msg)
        if self.sep not in key:
            if key not in self.data:
                self.data[key] = self._make_node()
            return self.data[key]  # type: ignore
        # need to create whole branch
        child, grandchildren = key.split(self.sep, 1)
        if child not in self.data:
            self.data[child] = self._make_node()
        return self.data[child].create_node(grandchildren)  # type: ignore

    def update_recursive(self, other: MutableMapping[str, Any]) -> None:
        """Recursively merges another mapping into this instance.

        Args:
            other (MutableMapping[str, Any]):
                Mapping whose entries are merged into this object. If both sides
                contain nested mappings at a key, values are merged recursively.
        """
        if not isinstance(other, MutableMapping):
            raise TypeError
        for k, v in other.items():
            if isinstance(v, MutableMapping):
                self.create_node(k).update_recursive(v)
            else:
                self[k] = v

    def update(self, other) -> None:  # type: ignore
        """Update this mapping from another mapping recursively.

        This method implements :class:`collections.abc.MutableMapping` update
        semantics for mapping-like inputs and forwards the actual merge to
        :meth:`update_recursive`.

        Args:
            other:
                Mapping containing keys and values to merge into this instance.

        Raises:
            TypeError:
                If `other` is not a mutable mapping.
        """
        self.update_recursive(other)

    def copy(self) -> TNestedDict:
        """Creates a structural copy with copied nested mappings.

        Child dictionaries and child `NestedDict` instances are copied, while
        non-mapping leaf values are reused by reference.

        Returns:
            NestedDict:
                New instance containing copied nested structure.
        """
        res = self._make_node()
        res.update_recursive(self)
        return res

    @overload
    def to_dict(self, *, flatten: Literal[False] = False) -> TDictTree: ...
    @overload
    def to_dict(self, *, flatten: Literal[True]) -> dict[str, TValue]: ...

    def to_dict(self, *, flatten: bool = False) -> TDictTree:  # type: ignore
        """Converts this object to a plain dictionary representation.

        Args:
            flatten (bool):
                If `True`, returns a flat mapping with separator-joined paths as
                keys. If `False`, returns nested dictionaries.

        Returns:
            dict[str, Any]:
                Dictionary representation of this object.

        Raises:
            TypeError:
                If flattening encounters non-string keys.
        """
        if flatten:
            return dict(self.items(flatten=True))
        # return hierarchical dictionaries
        return {
            k: v.to_dict(flatten=False) if isinstance(v, NestedDict) else v
            for k, v in self.data.items()
        }

    def pprint(self, *args: Any, flatten: bool = False, **kwargs: Any) -> None:
        """Pretty-prints the current data as nested plain dictionaries.

        Args:
            *args (Any):
                Positional arguments forwarded to `pprint.pprint`.
            flatten (bool):
                If `True`, returns a flat mapping with separator-joined paths as
                keys. If `False`, returns nested dictionaries.
            **kwargs (Any):
                Keyword arguments forwarded to `pprint.pprint`.
        """
        from pprint import pprint

        pprint(self.to_dict(flatten=flatten), *args, **kwargs)  # type: ignore
