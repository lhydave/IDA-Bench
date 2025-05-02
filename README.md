


Tianyu comment:

We may need to reinstall psutil. Run these commmands:

```sh
uv pip install --force-reinstall --no-binary psutil psutil
uv pip install ipykernel
python -m ipykernel install --user --name=myproject
```

User proxy:

I use tmux to keep the proxy running. Run the following command to start and attach to a session. 

```sh
tmux new-session -A -s proxy
```

In the session, run

```sh
uv pip intall litellm[proxy]
litellm --config llm_config_proxy.yaml
```
Press ``Ctrl+b``, then press ``d`` to detach from the session.

I have changed ``llm_config_user.toml`` and ``llm_config_agent.toml``. Simply run ``python llm_interact_tau_bench_debug.py``.

