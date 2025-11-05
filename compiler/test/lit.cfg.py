from lit.llvm import llvm_config
import lit.formats

config.name = "GraphAlg"

config.test_format = lit.formats.ShTest(execute_external=False)

config.suffixes = ['.mlir', '.gr']

config.test_source_root = os.path.dirname(__file__)

# Find common tools such as FileCheck
llvm_config.use_default_substitutions()

# The tools we want to use in lit test (inside RUN)
tools = [
    "graphalg-exec",
    "graphalg-opt",
    "graphalg-translate",
    "mlir-opt",
    "split-file",
]

# Where we look for the tools
tool_dirs = [
    config.llvm_tools_dir,
    config.graphalg_tools_dir,
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

config.environment["FILECHECK_OPTS"] = "--enable-var-scope --allow-unused-prefixes=false"
