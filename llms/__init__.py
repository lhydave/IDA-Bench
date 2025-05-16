from llms.user import User
from llms.shard_user import ShardUser
from llms.base_agent import BaseAgent


agent_dict = {"user": User, "base-agent": BaseAgent, "shard_user": ShardUser}
