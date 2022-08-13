# Documentation

To create the documentation, run the following command:

```bash
make -C docs html
sensible-browser docs/_build/html/index.html
make -C docs latexpdf
```
