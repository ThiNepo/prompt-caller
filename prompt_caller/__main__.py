import shutil
from pathlib import Path
from importlib.resources import as_file, files

import click


SKILL_NAME = "prompt-caller"


def _skill_source():
    return files("prompt_caller").joinpath("skills").joinpath(SKILL_NAME)


@click.group()
def cli():
    """PromptCaller command line interface."""


@cli.command()
@click.option(
    "--target",
    default=".agents/skills",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where skills are installed.",
)
def install(target: Path):
    """Install PromptCaller skills into .agents/skills."""
    source = _skill_source()
    if not source.is_dir():
        raise click.ClickException(
            f"Bundled skill '{SKILL_NAME}' was not found in package assets."
        )

    destination = target / SKILL_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)

    with as_file(source) as source_path:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source_path, destination)

    click.echo(f"Installed skill '{SKILL_NAME}' at: {destination}")


def main():
    cli()


if __name__ == "__main__":
    main()
