from llms.user import User
from llms.user2 import User2
from llms.base_agent import BaseAgent
from dataclasses import dataclass, asdict
import tomllib
from logger import logger
from typing import Any


agent_dict = {"user": User, "base-agent": BaseAgent, "user2": User2}
# TODO: Add more agent classes to this dictionary as needed

