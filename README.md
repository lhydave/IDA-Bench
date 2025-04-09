


Tianyu comment:

We may need to reinstall psutil. Run these commmands:

```sh
uv pip install --force-reinstall --no-binary psutil psutil
uv pip install ipykernel
python -m ipykernel install --user --name=myproject
```
