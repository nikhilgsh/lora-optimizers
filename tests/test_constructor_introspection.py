"""Focused tests for shared optimizer-constructor parameter discovery."""
from __future__ import annotations

from lora_playground.constructor_introspection import (
    forwardable_constructor_parameters,
)


class _Grandparent:
    def __init__(self, model, beyond_stop=11):
        pass


class _Consumer(_Grandparent):
    def __init__(self, targets, consumer_only=7, shared="consumer"):
        pass


class _ForwardingSubclass(_Consumer):
    def __init__(
        self,
        model,
        *args,
        lr=1e-3,
        leaf_only=3,
        shared="leaf",
        **kwargs,
    ):
        pass


for _class in (_Grandparent, _Consumer, _ForwardingSubclass):
    _class.__module__ = "lora_playground.test_constructor_introspection"


def test_forwarding_subclass_includes_inherited_parameters_leaf_first():
    parameters = forwardable_constructor_parameters(_ForwardingSubclass)

    assert [parameter.name for parameter in parameters] == [
        "leaf_only",
        "shared",
        "consumer_only",
    ]
    assert parameters[1].default == "leaf"


def test_walk_stops_at_first_constructor_without_var_keyword():
    names = {
        parameter.name
        for parameter in forwardable_constructor_parameters(_ForwardingSubclass)
    }

    assert "beyond_stop" not in names
    assert names.isdisjoint({"self", "model", "targets", "args", "kwargs", "lr"})


def test_optim_specs_preserves_existing_private_adapter_name():
    from lora_playground.optim_specs import _forwardable_params

    assert _forwardable_params is forwardable_constructor_parameters


def test_third_party_leaf_signature_is_discovered_without_walking_its_mro():
    from torch.optim import SGD

    names = {
        parameter.name for parameter in forwardable_constructor_parameters(SGD)
    }

    assert {"momentum", "weight_decay", "dampening", "nesterov"} <= names
    assert names.isdisjoint({"self", "params", "lr"})
