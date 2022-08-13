import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional

import frarch

here = Path(__file__).parent


@dataclass
class Function:
    module: str
    name: str


@dataclass
class Class:
    module: str
    name: str
    _functions: Optional[List[Function]] = None

    @property
    def functions(self) -> List[Function]:
        if self._functions is None:
            self._functions = self._load_functions()
        return self._functions

    def _load_functions(self) -> List[Function]:
        module = importlib.import_module(self.module)
        _class = getattr(module, self.name)
        module_functions = list(
            filter(
                lambda x: "frarch" in x[1].__module__ and not x[0].startswith("_"),
                inspect.getmembers(_class, inspect.isfunction),
            )
        )
        return [
            Function(module=self.module, name=function[0])
            for function in module_functions
        ]


@dataclass
class PackageFile:
    py_file: Path
    root_path: Path
    _classes: Optional[List[Class]] = None
    _functions: Optional[List[Function]] = None

    @property
    def module_name(self) -> str:
        return (
            str(self.py_file.relative_to(self.root_path))
            .replace("/", ".")
            .replace(".__init__", "")
            .replace(".py", "")
        )

    @property
    def classes(self) -> List[Class]:
        if self._classes is None:
            self._classes = self._load_classes()
        return self._classes

    def _load_classes(self) -> List[Class]:
        module = importlib.import_module(self.module_name)
        class_members = inspect.getmembers(module, inspect.isclass)
        module_class_members = [
            _class
            for _class in class_members
            if "frarch" in _class[1].__module__
            and self.module_name in _class[1].__module__
        ]
        return [
            Class(module=self.module_name, name=_class[0])
            for _class in module_class_members
        ]

    @property
    def functions(self) -> List[Function]:
        if self._functions is None:
            self._functions = self._load_functions()
        return self._functions

    def _load_functions(self) -> List[Function]:
        module = importlib.import_module(self.module_name)
        functions = inspect.getmembers(module, inspect.isfunction)
        module_functions = [
            function for function in functions if "frarch" in function[1].__module__
        ]
        return [
            Function(module=self.module_name, name=function[0])
            for function in module_functions
        ]

    @property
    def md_path(self) -> Path:
        return self.py_file.with_suffix(".md").relative_to(self.root_path)

    def generate_markdown_file(self):
        md_lines = [f"# {self.module_name}", ""]
        for _class in self.classes:
            md_lines.append(f"::: {_class.module}.{_class.name}")
            md_lines.append("")
        for function in self.functions:
            md_lines.append(f"::: {function.module}.{function.name}")
            md_lines.append("")

        md_abs_path = here / self.md_path
        md_abs_path.parent.mkdir(parents=True, exist_ok=True)
        md_abs_path.write_text("\n".join(md_lines))


def generate_reference(files: List[PackageFile]):
    md_paths = [file.md_path for file in files]
    lines = ["# Package reference", ""]
    for md_path in md_paths:
        module_path = str(md_path).replace("/__init__", "").replace(".md", "")
        module_name = module_path.split("/")[-1]
        indent = "  " * (len(module_path.split("/")) - 1)
        lines.append(f"{indent}- [{module_name}]({str(md_path)})")
    (here / "reference.md").write_text("\n".join(lines))


def main():
    package_path = Path(frarch.__file__).parent
    root_path = package_path.parent
    files: List[PackageFile] = [
        PackageFile(file, root_path) for file in package_path.glob("**/*.py")
    ]
    for file in files:
        file.generate_markdown_file()
    generate_reference(files)


if __name__ == "__main__":
    main()
