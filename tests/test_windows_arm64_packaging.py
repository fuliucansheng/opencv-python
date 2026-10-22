import ast
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("version,platform,machine,expected", [
    ("3.11", "win32", "ARM64", "==2.3.2"),
    ("3.12", "win32", "ARM64", "==2.3.2"),
    ("3.13", "win32", "ARM64", "==2.3.2"),
    ("3.14", "win32", "ARM64", "==2.3.2"),
    ("3.10", "win32", "ARM64", "==2.0.2"),
    ("3.13", "win32", "AMD64", "==2.1.3"),
    ("3.12", "win32", "AMD64", "==2.0.2"),
    ("3.13", "linux", "aarch64", "==2.1.3"),
    ("3.13", "darwin", "arm64", "==2.1.3"),
    ("3.14", "linux", "x86_64", "==2.3.2"),
])
def test_numpy_build_requirement(version, platform, machine, expected):
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    env = default_environment()
    env.update(python_version=version, sys_platform=platform, platform_machine=machine)
    requirements = [Requirement(value) for value in config["build-system"]["requires"]]
    numpy = [r for r in requirements if r.name == "numpy" and (r.marker is None or r.marker.evaluate(env))]
    assert len(numpy) == 1
    assert str(numpy[0].specifier) == expected

@pytest.mark.parametrize("os_name,platform,is64,expected", [
    ("nt", "win-arm64", True, []),
    ("nt", "win-amd64", True, [r"bin/opencv_videoio_ffmpeg\d{4}_64\.dll"]),
    ("nt", "win32", False, [r"bin/opencv_videoio_ffmpeg\d{4}\.dll"]),
    ("posix", "linux-aarch64", True, []),
    ("posix", "macosx-14.0-arm64", True, []),
])
def test_ffmpeg_staging_is_windows_architecture_specific(os_name, platform, is64, expected):
    filename = ROOT / "setup.py"
    module = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
    assignment = next(n for n in ast.walk(module) if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Name) and t.id == "rearrange_cmake_output_data" for t in n.targets))
    # Evaluate the real FFmpeg staging expression without invoking setup/build side effects.
    cv2_value = next(value for key, value in zip(assignment.value.keys, assignment.value.values) if key.value == "cv2")
    conditional = next(
        n for n in ast.walk(cv2_value)
        if isinstance(n, ast.IfExp) and any(
            isinstance(term, ast.Attribute) and term.attr == "name"
            and isinstance(term.value, ast.Name) and term.value.id == "os"
            for term in ast.walk(n.test)
        )
    )
    expression = compile(ast.Expression(conditional), str(filename), "eval")
    scope = {"os": SimpleNamespace(name=os_name), "sysconfig": SimpleNamespace(get_platform=lambda: platform), "is64": is64}
    assert eval(expression, scope) == expected
