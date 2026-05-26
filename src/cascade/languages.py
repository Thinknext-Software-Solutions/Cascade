"""Language profiles for multi-language code generation and testing.

Each LanguageProfile is a frozen bundle of conventions for one language:
where source/test files live, how to run tests, how to install deps, and
what marker files identify the language in a repo.

Adding a new language = add an entry to PROFILES. The rest of the pipeline
(planner, coder, tester) reads the profile and adapts.

Design rules:
- Profiles are immutable data, not classes with behavior
- Detection prefers specificity (tsconfig.json beats package.json for TS vs JS)
- Ambiguous repos (e.g. Python backend + React frontend) require explicit
  override in cascade.yaml; we never guess silently
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .exceptions import CascadeError


@dataclass(frozen=True)
class LanguageProfile:
    """Conventions and tooling for one programming language."""

    name: str  # canonical identifier, e.g. "python"
    display_name: str  # human label, e.g. "Python"
    file_extensions: tuple[str, ...]  # source file extensions including dot
    source_dir_default: str  # default location for source files
    test_dir_default: str  # default location for tests
    test_file_glob: str  # how test files are named
    test_command: tuple[str, ...]  # argv to run the test suite
    install_command: Optional[tuple[str, ...]]  # argv to install deps, or None
    type_check_command: Optional[tuple[str, ...]]  # argv to type-check, or None
    detection_files: tuple[str, ...]  # marker files indicating this language
    detection_priority: int = 50  # higher wins on ambiguity; 0-100
    formatter_command: Optional[tuple[str, ...]] = None  # optional format step
    notes_for_llm: str = ""  # passed to LLM prompts as language-specific guidance


# ----------------------------------------------------------------------------
# Built-in profiles
# ----------------------------------------------------------------------------


PYTHON = LanguageProfile(
    name="python",
    display_name="Python",
    file_extensions=(".py",),
    source_dir_default="src",
    test_dir_default="tests",
    test_file_glob="test_*.py",
    test_command=("pytest",),
    install_command=("pip", "install", "-e", ".[dev]"),
    type_check_command=("mypy", "src"),
    formatter_command=("black", "."),
    detection_files=("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"),
    detection_priority=60,
    notes_for_llm=(
        "Use type hints on all public function signatures. Prefer pathlib over "
        "os.path. Use pytest fixtures over setUp/tearDown. Tests live in tests/ "
        "and follow the test_*.py naming convention."
    ),
)


TYPESCRIPT = LanguageProfile(
    name="typescript",
    display_name="TypeScript",
    file_extensions=(".ts", ".tsx"),
    source_dir_default="src",
    test_dir_default="src",  # tests typically live alongside source in TS land
    test_file_glob="*.test.ts",
    test_command=("npx", "vitest", "run"),
    install_command=("npm", "install"),
    type_check_command=("npx", "tsc", "--noEmit"),
    formatter_command=("npx", "prettier", "--write", "."),
    detection_files=("tsconfig.json",),
    detection_priority=80,  # tsconfig is specific; wins over package.json alone
    notes_for_llm=(
        "Strict TypeScript: no `any`, prefer `unknown` then narrow. Use ES "
        "modules. Test files use the .test.ts suffix and import from vitest. "
        "Prefer named exports over default exports."
    ),
)


JAVASCRIPT = LanguageProfile(
    name="javascript",
    display_name="JavaScript",
    file_extensions=(".js", ".jsx", ".mjs"),
    source_dir_default="src",
    test_dir_default="src",
    test_file_glob="*.test.js",
    test_command=("npx", "vitest", "run"),
    install_command=("npm", "install"),
    type_check_command=None,
    formatter_command=("npx", "prettier", "--write", "."),
    detection_files=("package.json",),
    detection_priority=40,  # plain JS is the fallback; TS wins if tsconfig exists
    notes_for_llm=(
        "Use ES modules and modern syntax (const/let, arrow functions, async/await). "
        "JSDoc comments where types would help. Test files use the .test.js suffix."
    ),
)


GO = LanguageProfile(
    name="go",
    display_name="Go",
    file_extensions=(".go",),
    source_dir_default=".",  # Go's package layout is flat
    test_dir_default=".",  # _test.go files live next to source
    test_file_glob="*_test.go",
    test_command=("go", "test", "./..."),
    install_command=("go", "mod", "download"),
    type_check_command=("go", "vet", "./..."),
    formatter_command=("gofmt", "-w", "."),
    detection_files=("go.mod",),
    detection_priority=90,  # go.mod is highly specific
    notes_for_llm=(
        "Follow standard Go style: gofmt, no unused imports, error returns "
        "as the last return value. Test files end in _test.go and live in "
        "the same package as the code they test. Use the standard testing "
        "package; table-driven tests where appropriate."
    ),
)


RUST = LanguageProfile(
    name="rust",
    display_name="Rust",
    file_extensions=(".rs",),
    source_dir_default="src",
    test_dir_default="tests",  # integration tests; unit tests inline with mod tests
    test_file_glob="*.rs",
    test_command=("cargo", "test"),
    install_command=("cargo", "fetch"),
    type_check_command=("cargo", "check"),
    formatter_command=("cargo", "fmt"),
    detection_files=("Cargo.toml",),
    detection_priority=90,
    notes_for_llm=(
        "Idiomatic Rust: prefer Result over panics, use `?` for error "
        "propagation, derive common traits. Unit tests in #[cfg(test)] mod "
        "tests blocks within the same file. Integration tests in tests/."
    ),
)


JAVA = LanguageProfile(
    name="java",
    display_name="Java",
    file_extensions=(".java",),
    source_dir_default="src/main/java",
    test_dir_default="src/test/java",
    test_file_glob="*Test.java",
    test_command=("mvn", "test"),
    install_command=("mvn", "install", "-DskipTests"),
    type_check_command=None,  # the compiler is the type checker
    formatter_command=None,
    detection_files=("pom.xml", "build.gradle", "build.gradle.kts"),
    detection_priority=70,
    notes_for_llm=(
        "Java with JUnit 5. Test classes end in Test, in the parallel package "
        "under src/test/java. Use @Test, @BeforeEach, @ParameterizedTest where "
        "useful. Prefer constructor injection."
    ),
)


RUBY = LanguageProfile(
    name="ruby",
    display_name="Ruby",
    file_extensions=(".rb",),
    source_dir_default="lib",
    test_dir_default="spec",
    test_file_glob="*_spec.rb",
    test_command=("bundle", "exec", "rspec"),
    install_command=("bundle", "install"),
    type_check_command=None,
    formatter_command=("bundle", "exec", "rubocop", "-a"),
    detection_files=("Gemfile", "Rakefile"),
    detection_priority=70,
    notes_for_llm=(
        "Ruby with RSpec. Spec files end in _spec.rb under spec/. Use describe "
        "and it blocks. Prefer let over instance variables for setup."
    ),
)


CSHARP = LanguageProfile(
    name="csharp",
    display_name="C#",
    file_extensions=(".cs",),
    source_dir_default="src",
    test_dir_default="tests",
    test_file_glob="*Tests.cs",
    test_command=("dotnet", "test"),
    install_command=("dotnet", "restore"),
    type_check_command=("dotnet", "build", "--no-restore"),
    formatter_command=("dotnet", "format"),
    detection_files=("*.csproj", "*.sln"),  # glob handled specially
    detection_priority=70,
    notes_for_llm=(
        "C# with xUnit (preferred) or NUnit. Test class names end in Tests. "
        "Use [Fact] for single-case, [Theory] + [InlineData] for parameterized."
    ),
)


# Registry of all built-in profiles, keyed by name.
PROFILES: dict[str, LanguageProfile] = {
    p.name: p
    for p in (PYTHON, TYPESCRIPT, JAVASCRIPT, GO, RUST, JAVA, RUBY, CSHARP)
}


# ----------------------------------------------------------------------------
# Lookup and detection
# ----------------------------------------------------------------------------


def get_profile(name: str) -> LanguageProfile:
    """Return a LanguageProfile by canonical name.

    Args:
        name: A canonical language identifier (case-insensitive).

    Returns:
        The matching LanguageProfile.

    Raises:
        CascadeError: If no profile is registered with that name.
    """
    key = name.lower().strip()
    if key not in PROFILES:
        supported = ", ".join(sorted(PROFILES.keys()))
        raise CascadeError(
            f"Unknown language '{name}'. Supported in v0.1: {supported}."
        )
    return PROFILES[key]


def detect_language(repo_root: Path) -> Optional[LanguageProfile]:
    """Detect the primary language of a repo by looking for marker files.

    Picks the highest-priority profile whose detection_files exist in
    `repo_root`. Returns None if nothing matches.

    Glob patterns in `detection_files` (containing '*') are handled by
    Path.glob; literal names are checked directly. Detection is non-recursive
    -- we only look at the repo root, not subdirectories.

    Args:
        repo_root: Path to the repository root.

    Returns:
        The detected LanguageProfile or None if nothing matched.
    """
    if not repo_root.is_dir():
        return None

    candidates: list[LanguageProfile] = []
    for profile in PROFILES.values():
        for pattern in profile.detection_files:
            if "*" in pattern:
                if any(repo_root.glob(pattern)):
                    candidates.append(profile)
                    break
            else:
                if (repo_root / pattern).exists():
                    candidates.append(profile)
                    break

    if not candidates:
        return None

    # Highest detection_priority wins; ties broken alphabetically by name.
    candidates.sort(key=lambda p: (-p.detection_priority, p.name))
    return candidates[0]


def resolve_language(
    repo_root: Path, configured_name: Optional[str] = None
) -> LanguageProfile:
    """Resolve which language profile to use for a repo.

    Order of precedence:
        1. Explicit configured_name (from cascade.yaml or CLI flag)
        2. Auto-detected from marker files in repo_root
        3. Raise CascadeError if neither yields a profile

    Args:
        repo_root: Repository root.
        configured_name: Optional explicit override.

    Returns:
        A LanguageProfile.

    Raises:
        CascadeError: If no language can be resolved.
    """
    if configured_name:
        return get_profile(configured_name)
    detected = detect_language(repo_root)
    if detected is not None:
        return detected
    supported = ", ".join(sorted(PROFILES.keys()))
    raise CascadeError(
        "Could not detect the project's language",
        hint=[
            "Set it in cascade.yaml:\n"
            "language: python   # or typescript, go, rust, java, ruby, csharp, javascript",
            "Or pass it on the CLI: --language python",
            f"Supported: {supported}",
            "Cascade detects from marker files (pyproject.toml, package.json, go.mod, etc). "
            "If none are at the repo root, configure explicitly.",
        ],
        learn_more="cascade doctor",
    )
