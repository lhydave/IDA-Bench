


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

User proxy:

I use tmux to keep the proxy running. Run the following command to start and attach to a session. 

```sh
tmux new-session -A -s proxy
```

In the session, first activate the correct virtual environment if we are in `base`.

```sh
source .venv/bin/activate
```
then run
```sh
uv pip install litellm[proxy]
litellm --config llm_configs/llm_config_proxy.yaml
```
Press ``Ctrl+b`` (press those two keys together), then release your fingers and press ``d`` to detach from the session.
To return to this session, use
```sh
tmux a -t proxy
```

I have changed ``llm_config_user.toml`` and ``llm_config_agent.toml``. Simply run ``python llm_interact_tau_bench_debug.py``.


Trajactory to markdown:

```sh
python utils/trajactory_to_markdown_new.py --input 
```