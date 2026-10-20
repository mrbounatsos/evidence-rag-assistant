from pathlib import Path


def test_project_readme_exists() -> None:
    assert Path("README.md").is_file()
