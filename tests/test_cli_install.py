from pathlib import Path
import subprocess
import sys
import shutil

from click.testing import CliRunner
import pytest

from prompt_caller.__main__ import SKILL_NAME, _skill_source, cli


def test_bundled_skill_source_exists():
    source = _skill_source()
    assert source.is_dir()
    assert source.joinpath("SKILL.md").is_file()
    assert source.joinpath("agents", "openai.yaml").is_file()
    assert source.joinpath("references", "prompt-patterns.md").is_file()


@pytest.fixture
def cli_runtime_root() -> Path:
    root = Path(".test_runtime_cli")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_install_uses_default_agents_skills_path(cli_runtime_root, monkeypatch):
    workdir = (cli_runtime_root / "default").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)
    runner = CliRunner()
    result = runner.invoke(cli, ["install"])

    assert result.exit_code == 0
    destination = workdir / ".agents" / "skills" / SKILL_NAME
    assert destination.exists()
    assert (destination / "SKILL.md").exists()


def test_install_overwrites_existing_skill_directory(cli_runtime_root, monkeypatch):
    workdir = (cli_runtime_root / "overwrite").resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    target_root = workdir / ".agents" / "skills"
    existing = target_root / SKILL_NAME
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "SKILL.md").write_text("old content", encoding="utf-8")

    monkeypatch.chdir(workdir)
    runner = CliRunner()
    result = runner.invoke(cli, ["install"])

    assert result.exit_code == 0
    installed_content = (existing / "SKILL.md").read_text(encoding="utf-8")
    assert "name: prompt-caller" in installed_content
    assert "old content" not in installed_content


def test_install_accepts_custom_target(cli_runtime_root):
    target_root = cli_runtime_root / "custom" / "my-skills"
    runner = CliRunner()
    result = runner.invoke(cli, ["install", "--target", str(target_root)])

    assert result.exit_code == 0
    assert (target_root / SKILL_NAME / "SKILL.md").exists()


def test_module_invocation_installs_skill(cli_runtime_root):
    target_root = cli_runtime_root / "module" / "module-skills"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "prompt_caller",
            "install",
            "--target",
            str(target_root),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (target_root / SKILL_NAME / "SKILL.md").exists()


def test_setup_console_entrypoint_is_mapped_to_cli():
    setup_text = (Path(__file__).resolve().parents[1] / "setup.py").read_text(
        encoding="utf-8"
    )
    assert "prompt-caller=prompt_caller.__main__:cli" in setup_text
