# Test

To run the tests for the frarch package:

```bash
python -m unittest discover
```

or with coverage

```bash
coverage run -m unittest discover -s tests/unit
coverage report -m --omit='tests/unit/*'
```
