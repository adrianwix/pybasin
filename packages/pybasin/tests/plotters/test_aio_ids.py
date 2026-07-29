"""Tests for AIO pattern-matching ID utilities."""

from pybasin.plotters.interactive_plotter.ids_aio import aio_id


def test_aio_id_structure():
    """Test that aio_id returns correctly structured dictionary."""
    result = aio_id("StateSpace", "test-123", "plot")

    assert isinstance(result, dict)
    assert "component" in result
    assert "aio_id" in result
    assert "subcomponent" in result
    assert result["component"] == "StateSpace"
    assert result["aio_id"] == "test-123"
    assert result["subcomponent"] == "plot"


def test_aio_id_uniqueness():
    """Test that different parameters produce different IDs."""
    id1 = aio_id("StateSpace", "instance-1", "plot")
    id2 = aio_id("StateSpace", "instance-2", "plot")
    id3 = aio_id("FeatureSpace", "instance-1", "plot")
    id4 = aio_id("StateSpace", "instance-1", "controls")

    assert id1 != id2
    assert id1 != id3
    assert id1 != id4
    assert id2 != id3


def test_aio_id_with_special_characters():
    """Test aio_id handles special characters in parameters."""
    result = aio_id("Component-Type", "uuid-1234-5678", "sub_component")

    assert result["component"] == "Component-Type"
    assert result["aio_id"] == "uuid-1234-5678"
    assert result["subcomponent"] == "sub_component"


def test_aio_id_with_numbers():
    """Test aio_id handles numeric instance IDs."""
    result = aio_id("ParamPage", "5", "plot")

    assert result["component"] == "ParamPage"
    assert result["aio_id"] == "5"
    assert result["subcomponent"] == "plot"
