# `name` is the name of the package as used for `pip install package`
name = "frarch"
# `path` is the name of the package for `import package`
path = name.lower().replace("-", "_").replace(" ", "_")
# Version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
version = "0.0.1"
author = "vbadenas"
author_email = "victor.badenas@gmail.com"
description = "Training Framework for PyTorch projects"  # One-liner
url = "https://github.com/vbadenas/frarch"  # your project homepage
license = "MIT"  # See https://choosealicense.com
