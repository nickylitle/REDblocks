"""Build hook: the editor's examples (examples/learn and the Apollo 11 showcase) live outside the package in the source
tree. Every build (pip install ., a wheel, build_exes.bat) copies them into simblocks/app/examples inside the build,
so an installed app always has its Examples menu, Help example links and the showcase."""
import glob
import os
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
COPIES = [("examples/learn/*.yaml", "simblocks/app/examples"),
          ("examples/learn/data/*.csv", "simblocks/app/examples/data"),
          ("examples/showcase/apollo11/*.yaml", "simblocks/app/examples/showcase"),
          ("examples/showcase/apollo11/apollo11.html", "simblocks/app/examples/showcase")]


class BuildWithExamples(build_py):
    def run(self):
        super().run()
        for pattern, dest in COPIES:
            out = os.path.join(self.build_lib, dest)
            os.makedirs(out, exist_ok=True)
            for f in glob.glob(os.path.join(HERE, pattern)):
                shutil.copy2(f, out)


setup(cmdclass={"build_py": BuildWithExamples})
