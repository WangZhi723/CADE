"""
训练检查点：保存/加载网络与优化器状态、指标与随机数状态，支持断点续训与 WandB resume。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, list):
            out[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _agent_kind(agent) -> str:
    from algorithms import DDPG, PPOAgent, DR3L

    if isinstance(agent, DDPG):
        return "ddpg"
    if isinstance(agent, PPOAgent):
        return "ppo"
    if isinstance(agent, DR3L):
        return "dr3l"
    raise TypeError(f"Unsupported agent type: {type(agent)}")


def _collect_rng_state(device: str) -> Dict[str, Any]:
    st: Dict[str, Any] = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device == "cuda" and torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng_state(st: Dict[str, Any], device: str) -> None:
    if "numpy" in st:
        np.random.set_state(st["numpy"])
    if "torch" in st:
        torch.set_rng_state(st["torch"])
    if device == "cuda" and torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def save_agent_state(agent, kind: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    nets: Dict[str, Any] = {}
    optim: Dict[str, Any] = {}
    if kind == "ddpg":
        nets = {
            "actor": agent.actor.state_dict(),
            "actor_target": agent.actor_target.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
        }
        optim = {
            "actor_optimizer": agent.actor_optimizer.state_dict(),
            "critic_optimizer": agent.critic_optimizer.state_dict(),
        }
    elif kind == "ppo":
        nets = {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "log_std": agent.log_std.detach().cpu(),
        }
        optim = {"optimizer": agent.optimizer.state_dict()}
    elif kind == "dr3l":
        nets = {
            "feature_net": agent.feature_net.state_dict(),
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
        }
        optim = {
            "optimizer_actor": agent.optimizer_actor.state_dict(),
            "optimizer_feature_critic": agent.optimizer_feature_critic.state_dict(),
        }
    return nets, optim


def load_agent_state(agent, kind: str, nets: Dict[str, Any], optim: Dict[str, Any]) -> None:
    if kind == "ddpg":
        agent.actor.load_state_dict(nets["actor"])
        agent.actor_target.load_state_dict(nets["actor_target"])
        agent.critic.load_state_dict(nets["critic"])
        agent.critic_target.load_state_dict(nets["critic_target"])
        agent.actor_optimizer.load_state_dict(optim["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(optim["critic_optimizer"])
    elif kind == "ppo":
        agent.actor.load_state_dict(nets["actor"])
        agent.critic.load_state_dict(nets["critic"])
        agent.log_std.data.copy_(nets["log_std"].to(agent.log_std.device))
        agent.optimizer.load_state_dict(optim["optimizer"])
    elif kind == "dr3l":
        agent.feature_net.load_state_dict(nets["feature_net"])
        agent.actor.load_state_dict(nets["actor"])
        agent.critic.load_state_dict(nets["critic"])
        agent.critic_target.load_state_dict(nets["critic_target"])
        agent.optimizer_actor.load_state_dict(optim["optimizer_actor"])
        if "optimizer_feature_critic" in optim:
            agent.optimizer_feature_critic.load_state_dict(optim["optimizer_feature_critic"])
        else:
            agent.optimizer_feature_critic.load_state_dict(optim["optimizer_critic"])


def save_training_checkpoint(
    agent,
    checkpoint_dir: Path,
    last_completed_episode: int,
    metrics: Dict[str, List[Any]],
    wandb_run_id: Optional[str],
    device: str,
    extra: Optional[Dict[str, List[Any]]] = None,
) -> None:
    """last_completed_episode 为已完成的最后一个 episode 索引（0-based）。"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    kind = _agent_kind(agent)
    nets, optim = save_agent_state(agent, kind)
    payload = {
        "kind": kind,
        "nets": nets,
        "optimizers": optim,
        "rng": _collect_rng_state(device),
    }
    torch.save(payload, checkpoint_dir / "agent.pt")

    meta = {
        "last_completed_episode": int(last_completed_episode),
        "metrics": _serialize_metrics(metrics),
        "wandb_run_id": wandb_run_id,
        "agent_kind": kind,
    }
    if extra:
        meta["extra"] = _serialize_metrics(extra)
    with open(checkpoint_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_training_checkpoint(
    agent, checkpoint_dir: Path, device: str
) -> Tuple[int, Dict[str, List[Any]], Optional[str], Dict[str, List[Any]]]:
    """
    返回：(下一轮应开始的 episode 索引, metrics 字典, wandb_run_id, extra)
    """
    checkpoint_dir = Path(checkpoint_dir)
    payload = torch.load(checkpoint_dir / "agent.pt", map_location=device)
    with open(checkpoint_dir / "training_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    kind = payload["kind"]
    load_agent_state(agent, kind, payload["nets"], payload["optimizers"])
    _restore_rng_state(payload["rng"], device)

    last = int(meta["last_completed_episode"])
    metrics = {k: list(v) for k, v in meta.get("metrics", {}).items()}
    extra = {k: list(v) for k, v in meta.get("extra", {}).items()}
    wid = meta.get("wandb_run_id")
    return last + 1, metrics, wid, extra


def checkpoint_exists(checkpoint_dir: Path) -> bool:
    d = Path(checkpoint_dir)
    return (d / "agent.pt").is_file() and (d / "training_meta.json").is_file()
