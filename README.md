


Tianyu comment:

We may need to reinstall psutil. Run these commmands:

```sh
uv pip install --force-reinstall --no-binary psutil psutil
uv pip install ipykernel
python -m ipykernel install --user --name=myproject
uv pip install matplotlib
uv pip install pandas
uv pip install seaborn
uv pip install statsmodels
uv pip install scikit-learn
uv pip install open-interpreter
```
