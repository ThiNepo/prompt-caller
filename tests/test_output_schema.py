from typing import Literal, get_args, get_origin

import pytest


def test_legacy_output_defaults_to_string_fields(prompt_caller):
    Model = prompt_caller.createPydanticModel(
        {
            "result": "Final result",
            "explanation": "Short explanation",
        }
    )

    assert Model.model_fields["result"].annotation is str
    assert Model.model_fields["explanation"].annotation is str
    assert Model.model_fields["result"].description == "Final result"


def test_compact_dsl_supports_types_lists_enums_optionals(prompt_caller):
    Model = prompt_caller.createPydanticModel(
        {
            "result": "number | Final value",
            "confidence": "enum[low|medium|high] | Confidence level",
            "steps": "list[Step] | Ordered steps",
            "note?": "string | Optional note",
        },
        type_defs={
            "Step": {
                "expression": "string | expression",
                "value": "number | value",
            }
        },
    )

    assert Model.model_fields["result"].annotation is float
    assert get_origin(Model.model_fields["steps"].annotation) is list

    enum_annotation = Model.model_fields["confidence"].annotation
    assert get_origin(enum_annotation) is Literal
    assert get_args(enum_annotation) == ("low", "medium", "high")

    optional_note = Model.model_fields["note"].annotation
    assert get_origin(optional_note) is not None

    instance = Model(
        result=10.2,
        confidence="high",
        steps=[{"expression": "1+1", "value": 2}],
    )
    assert instance.note is None


def test_enum_descriptions_are_merged_into_field_description(prompt_caller):
    Model = prompt_caller.createPydanticModel(
        {
            "confidence": {
                "type": "enum[low|high]",
                "description": "Model confidence",
                "enum_descriptions": {
                    "low": "uncertain",
                    "high": "verified",
                },
            }
        }
    )

    description = Model.model_fields["confidence"].description
    assert "Model confidence" in description
    assert "low=uncertain" in description
    assert "high=verified" in description


@pytest.mark.parametrize(
    "schema,error_text",
    [
        ({"items": "list[] | bad"}, "list type requires inner type"),
        ({"score": "UnknownType | bad"}, "unknown type expression"),
        ({"a?b": "string | bad"}, r"optional marker '\?' is only allowed"),
        (
            {
                "confidence": {
                    "type": "enum[low|high]",
                    "enum_descriptions": {"medium": "invalid"},
                }
            },
            "unknown values",
        ),
    ],
)
def test_invalid_output_schema_raises_clear_errors(prompt_caller, schema, error_text):
    with pytest.raises(ValueError, match=error_text):
        prompt_caller.createPydanticModel(schema)
