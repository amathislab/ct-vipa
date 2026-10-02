"""
User-defined label classes for the three annotation categories (species, activities, actions).
Classes are declared in the `labels` section of the config file, and CVAT attributes are mapped to these categories.
Activities: High-level behaviors
Actions: Low-level behaviors, up to two simulateneous actions allowed.
This util script can be extended to account for more categories (e.g., animal sex or age)
"""

from typing import Dict, List

from data_processing.utils.db_utils import get_parameters_from_config

# CVAT attribute name -> label category in the config
LABEL_ATTRIBUTES: Dict[str, str] = {
    "Species": "species",
    "Activity": "activities",
    "Action": "actions",
    "Action2": "actions",
}

# Value of the secondary action when there is none
NO_SECONDARY_ACTION: str = "none"

# Attributes that cannot change along a tracklet
IMMUTABLE_ATTRIBUTES: List[str] = ["Species"]


def get_label_classes(config: Dict) -> Dict[str, List[str]]:
    return get_parameters_from_config(config, "labels")


def allowed_values(label_classes: Dict[str, List[str]], attribute: str) -> List[str]:
    """
    Returns the values a CVAT attribute can take; the first value is the default one.
    """
    values = list(label_classes[LABEL_ATTRIBUTES[attribute]])
    if attribute == "Action2":
        values = [NO_SECONDARY_ACTION] + values
    return values


def build_cvat_label_constructor(label_classes: Dict[str, List[str]]) -> List[Dict]:
    """
    Builds the CVAT raw label constructor for tracklet annotation.
    """
    attributes = []
    for attribute_id, attribute in enumerate(LABEL_ATTRIBUTES, start=1):
        values = allowed_values(label_classes, attribute)
        attributes.append(
            {
                "id": attribute_id,
                "name": attribute,
                "input_type": "select",
                "mutable": attribute not in IMMUTABLE_ATTRIBUTES,
                "values": values,
                "default_value": values[0],
            }
        )

    return [
        {
            "name": "Animal",
            "id": 1,
            "color": "#cc3366",
            "type": "rectangle",
            "attributes": attributes,
        }
    ]


def find_unknown_labels(
    attributes: Dict[str, str], label_classes: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Returns the label attributes whose value is not among the configured classes.
    """
    return {
        attribute: attributes[attribute]
        for attribute in LABEL_ATTRIBUTES
        if attribute in attributes
        and attributes[attribute] not in allowed_values(label_classes, attribute)
    }
