from prompt_caller import PromptCaller

ai = PromptCaller()

import os
import base64


def load_pdf() -> str:
    """
    Load pdf
    """

    root_path = os.path.join("model.pdf")

    with open(root_path, "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

    return [
        {
            "type": "input_file",
            "filename": "model.pdf",
            "file_data": f"data:application/pdf;base64,{pdf_base64}",
        },
    ]


response = ai.agent(
    "sample-reading-pdf",
    tools=[load_pdf],
    # output=ResponseFormatter,
)

print(response)
