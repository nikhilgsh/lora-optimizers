from lora_playground.train import (
    format_example,
    format_example_with_boundary,
    parse_target_modules,
)


def test_format_prompt_completion_example():
    text = format_example({"prompt": "Write Python.", "completion": "print('ok')"})
    assert text == "Write Python.\nprint('ok')"


def test_format_alpaca_example_with_input():
    text = format_example(
        {
            "instruction": "Add two numbers.",
            "input": "1 2",
            "output": "3",
        }
    )
    assert "Instruction:" in text
    assert "Input:" in text
    assert "Response:" in text
    assert text.endswith("3")


def test_format_instruction_response_example():
    text = format_example({"instruction": "Square x.", "response": "return x * x"})
    assert text == "Instruction:\nSquare x.\n\nResponse:\nreturn x * x"


def test_format_instruction_output_no_input():
    """OpenCoder opc-sft-stage2 schema: {instruction, output} with no input."""
    ex = {"instruction": "Reverse the list.", "output": "lst[::-1]"}
    text = format_example(ex)
    assert "Instruction:\nReverse the list." in text
    assert "Input:" not in text
    assert text.endswith("Response:\nlst[::-1]")
    prompt, response = format_example_with_boundary(ex)
    assert prompt == "Instruction:\nReverse the list.\n\nResponse:\n"
    assert response == "lst[::-1]"


def test_format_instruction_output_empty_input():
    """Defensive: input field present but empty should be skipped."""
    ex = {"instruction": "Reverse the list.", "input": "", "output": "lst[::-1]"}
    text = format_example(ex)
    assert "Input:" not in text
    prompt, response = format_example_with_boundary(ex)
    assert "Input:" not in prompt


def test_parse_target_modules():
    assert parse_target_modules("all-linear") == "all-linear"
    assert parse_target_modules("q_proj,k_proj, v_proj") == ["q_proj", "k_proj", "v_proj"]
