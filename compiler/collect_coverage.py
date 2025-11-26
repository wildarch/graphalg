#!/usr/bin/env python3
import subprocess
import sys
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).parent
BUILD_DIR = WORKSPACE_ROOT / "build-coverage"


def main():
    # Run configure-coverage.sh only if build dir does not exist yet
    if not BUILD_DIR.exists():
        configure_script = WORKSPACE_ROOT / "configure-coverage.sh"
        subprocess.run([configure_script], check=True)
    else:
        # If build dir already exists, delete coverage directories
        coverage_dir = BUILD_DIR / "test" / "coverage"
        coverage_report_dir = BUILD_DIR / "coverage-report"

        if coverage_dir.exists():
            shutil.rmtree(coverage_dir)
        if coverage_report_dir.exists():
            shutil.rmtree(coverage_report_dir)

    # Build with cmake
    subprocess.run(["cmake", "--build", BUILD_DIR, "--target", "check"], check=True)

    # Merge coverage data
    coverage_dir = BUILD_DIR / "test" / "coverage"
    profraw_files = list(coverage_dir.glob("*.profraw"))

    if not profraw_files:
        print(f"Error: No .profraw files found in {coverage_dir}", file=sys.stderr)
        sys.exit(1)

    profdata_output = BUILD_DIR / "compiler.profdata"
    subprocess.run([
        "llvm-profdata-20", "merge", "-sparse",
        *profraw_files,
        "-o", profdata_output
    ], check=True)

    # Generate coverage reports for all libraries
    coverage_report_base = BUILD_DIR / "coverage-report"
    coverage_report_base.mkdir(exist_ok=True)

    # Find all .a library files in the build directory
    library_files = list(BUILD_DIR.glob("**/*.a"))

    if not library_files:
        print("Error: No .a library files found in build directory", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(library_files)} libraries, generating coverage reports...")

    for lib_file in library_files:
        # Create a report directory based on the library name
        lib_name = lib_file.stem  # e.g., "libGraphAlgIR" -> "GraphAlgIR"
        if lib_name.startswith("lib"):
            lib_name = lib_name[3:]  # Remove "lib" prefix

        report_dir = coverage_report_base / lib_name
        report_dir.mkdir(exist_ok=True)

        print(f"  Generating report for {lib_file.name}...")
        subprocess.run([
            "llvm-cov-20", "show",
            lib_file,
            f"-instr-profile={profdata_output}",
            "--ignore-filename-regex=opt/llvm-debug",
            "--format=html",
            "-o", report_dir
        ], check=True)

    print(f"\nCoverage reports generated at {coverage_report_base}")


if __name__ == "__main__":
    main()
