"""System-file compiler: YAML model -> verified sim -> report, with no hand-written loop.
    from simblocks.compiler import load_system
    sysm = load_system("model.yaml"); out = sysm.run(); rep = sysm.verify(); sysm.report("out/")"""
from .loader import load, check, SpecError
from .system import System, load_system
