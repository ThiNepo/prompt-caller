from prompt_caller import PromptCaller

ai = PromptCaller()

import os
import base64


def image_to_base64() -> str:
    """
    Transform image to base64 to be used by the call
    """

    root_path = os.path.join("image.jpeg")

    with open(root_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}",
            },
        }
    ]


response = ai.agent(
    "sample-reading-image",
    tools=[image_to_base64],
    # output=ResponseFormatter,
)

print(response)
