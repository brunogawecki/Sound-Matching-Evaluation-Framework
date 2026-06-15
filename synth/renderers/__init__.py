from .base import Renderer


def make_renderer(name: str, plugin_path: str, sample_rate: int, buffer_size: int) -> Renderer:
    """
    Construct a Renderer by short name.

    Args:
        name: 'dawdreamer' or 'pedalboard'.
        plugin_path: Path to the VST3 plugin.
        sample_rate: Render sample rate in Hz.
        buffer_size: Engine block size (used by DawDreamer; ignored by Pedalboard's API).

    Raises:
        ValueError: If the renderer name is unknown.
    """
    if name == "dawdreamer":
        from .dawdreamer_renderer import DawDreamerRenderer
        return DawDreamerRenderer(plugin_path, sample_rate, buffer_size)
    if name == "pedalboard":
        from .pedalboard_renderer import PedalboardRenderer
        return PedalboardRenderer(plugin_path, sample_rate)
    raise ValueError(f"Unknown renderer '{name}'. Expected 'dawdreamer' or 'pedalboard'.")


__all__ = ["Renderer", "make_renderer"]
