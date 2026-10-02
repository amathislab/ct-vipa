"""
Unit tests for data_processing/utils/label_utils.py and the label corrections of
data_processing/process_tracklet_annotations_from_cvat.py

Tests label handling including:
- Allowed values of each CVAT attribute
- CVAT label constructor generation
- Detection of values outside the configured classes
- Action corrections applied to CVAT annotations
"""

from typing import Dict, List

import pytest

from data_processing.process_tracklet_annotations_from_cvat import correct_detection
from data_processing.utils.label_utils import (
    LABEL_ATTRIBUTES,
    NO_SECONDARY_ACTION,
    allowed_values,
    build_cvat_label_constructor,
    find_unknown_labels,
    get_label_classes,
)


@pytest.fixture
def label_classes() -> Dict[str, List[str]]:
    return {
        "species": ["fox", "hare", "unknown"],
        "activities": ["foraging", "resting"],
        "actions": ["walking", "standing_head_up", "vocalizing"],
    }


class TestGetLabelClasses:
    """Test cases for the get_label_classes function."""

    def test_returns_labels_section(self, label_classes):
        assert get_label_classes({"labels": label_classes}) == label_classes

    def test_missing_labels_section_raises(self):
        with pytest.raises(ValueError):
            get_label_classes({"global": {}})


class TestAllowedValues:
    """Test cases for the allowed_values function."""

    def test_category_classes(self, label_classes):
        assert allowed_values(label_classes, "Species") == ["fox", "hare", "unknown"]
        assert allowed_values(label_classes, "Activity") == ["foraging", "resting"]
        assert allowed_values(label_classes, "Action") == label_classes["actions"]

    def test_secondary_action_starts_with_none(self, label_classes):
        assert (
            allowed_values(label_classes, "Action2")
            == [NO_SECONDARY_ACTION] + label_classes["actions"]
        )

    def test_does_not_modify_config(self, label_classes):
        allowed_values(label_classes, "Action2")
        assert NO_SECONDARY_ACTION not in label_classes["actions"]


class TestBuildCvatLabelConstructor:
    """Test cases for the build_cvat_label_constructor function."""

    def test_single_animal_label(self, label_classes):
        constructor = build_cvat_label_constructor(label_classes)
        assert len(constructor) == 1
        assert constructor[0]["name"] == "Animal"
        assert constructor[0]["type"] == "rectangle"

    def test_one_select_attribute_per_label_attribute(self, label_classes):
        attributes = build_cvat_label_constructor(label_classes)[0]["attributes"]
        assert [a["name"] for a in attributes] == list(LABEL_ATTRIBUTES)
        assert all(a["input_type"] == "select" for a in attributes)
        assert len({a["id"] for a in attributes}) == len(attributes)

    def test_values_and_defaults(self, label_classes):
        attributes = {
            a["name"]: a
            for a in build_cvat_label_constructor(label_classes)[0]["attributes"]
        }
        for name, attribute in attributes.items():
            assert attribute["values"] == allowed_values(label_classes, name)
            assert attribute["default_value"] == attribute["values"][0]
        assert attributes["Action2"]["default_value"] == NO_SECONDARY_ACTION

    def test_species_is_immutable(self, label_classes):
        attributes = {
            a["name"]: a
            for a in build_cvat_label_constructor(label_classes)[0]["attributes"]
        }
        assert attributes["Species"]["mutable"] is False
        assert attributes["Action"]["mutable"] is True


class TestFindUnknownLabels:
    """Test cases for the find_unknown_labels function."""

    def test_all_known(self, label_classes):
        attributes = {
            "Species": "fox",
            "Activity": "resting",
            "Action": "walking",
            "Action2": NO_SECONDARY_ACTION,
        }
        assert find_unknown_labels(attributes, label_classes) == {}

    def test_unknown_values(self, label_classes):
        attributes = {
            "Species": "wolf",
            "Activity": "resting",
            "Action": "running",
            "Action2": NO_SECONDARY_ACTION,
        }
        assert find_unknown_labels(attributes, label_classes) == {
            "Species": "wolf",
            "Action": "running",
        }

    def test_none_is_only_valid_for_secondary_action(self, label_classes):
        attributes = {"Action": NO_SECONDARY_ACTION}
        assert find_unknown_labels(attributes, label_classes) == {
            "Action": NO_SECONDARY_ACTION
        }

    def test_ignores_other_and_missing_attributes(self, label_classes):
        attributes = {"Species": "fox", "Deer_age": "juvenile"}
        assert find_unknown_labels(attributes, label_classes) == {}


class TestCorrectDetection:
    """Test cases for the correct_detection function of the CVAT tracklet annotations stage."""

    @staticmethod
    def make_detection(**attributes) -> Dict:
        default_attributes = {
            "Species": "fox",
            "Activity": "foraging",
            "Action": "walking",
            "Action2": NO_SECONDARY_ACTION,
        }
        return {"attributes": {**default_attributes, **attributes}}

    def test_moves_secondary_actions(self):
        detection = correct_detection(
            self.make_detection(Action="vocalizing"),
            always_second_action=["vocalizing"],
            default_first_action="standing_head_up",
        )
        assert detection["attributes"]["Action"] == "standing_head_up"
        assert detection["attributes"]["Action2"] == "vocalizing"

    def test_removes_duplicate_secondary_action(self):
        detection = correct_detection(
            self.make_detection(Action="walking", Action2="walking"),
            always_second_action=[],
            default_first_action="standing_head_up",
        )
        assert detection["attributes"]["Action"] == "walking"
        assert detection["attributes"]["Action2"] == NO_SECONDARY_ACTION

    def test_keeps_other_attributes(self):
        detection = correct_detection(
            self.make_detection(Species="fox", Deer_age="adult", Deer_adult_sex="male"),
            always_second_action=[],
            default_first_action="standing_head_up",
        )
        assert detection["attributes"]["Deer_age"] == "adult"
        assert detection["attributes"]["Deer_adult_sex"] == "male"
