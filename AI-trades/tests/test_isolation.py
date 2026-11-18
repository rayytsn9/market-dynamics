from pathlib import Path


def test_experiment_package_does_not_import_research_modules():
    package = Path(__file__).resolve().parents[1] / "ai_trades"
    for path in package.glob("*.py"):
        text = path.read_text()
        assert "import src" not in text
        assert "from src" not in text
