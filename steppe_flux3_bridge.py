"""Optional adapter for the installed Project Invisible FLUX.3 runtime.

Only bridge.generate stays wrapped. Decoder and engine hooks are temporary,
thread-local and removed even when generation fails. No FLUX.3 files change.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import importlib
import threading

import torch

_lock = threading.RLock()
_active = ContextVar("steppe_flux3_request", default=None)
_request_callback = ContextVar("steppe_flux3_callback", default=None)
_installed = None
_installed_owner = None
_installed_original = None
_installation_state = None


def upscale_tokens(tokens, ids, callback, seed):
    """Translate official BN-normalized token grids to/from Steppe BCHW."""
    if tokens.ndim != 2 or tokens.shape[1] != 128:
        raise ValueError("Steppe FLUX.3 requires 128-channel normalized tokens")
    if ids.ndim != 2 or ids.shape != (tokens.shape[0], 4):
        raise ValueError("Steppe FLUX.3 requires four-coordinate token IDs")
    if not torch.isfinite(ids).all() or not torch.equal(ids, ids.round()):
        raise ValueError("Steppe FLUX.3 requires integer grid IDs")
    if (ids[:, (0, 3)] != 0).any() or (ids[:, 1:3] < 0).any():
        raise ValueError("Steppe FLUX.3 supports a single image grid")
    h, w = int(ids[:, 1].max()) + 1, int(ids[:, 2].max()) + 1
    positions = (ids[:, 1].long() * w + ids[:, 2].long()).to(tokens.device)
    if h * w != tokens.shape[0] or positions.unique().numel() != h * w:
        raise ValueError("Steppe FLUX.3 token grid has holes or repeated IDs")
    grid = torch.empty_like(tokens)
    grid[positions] = tokens
    packed = grid.T.reshape(1, 128, h, w)
    result = callback(packed, int(seed))
    if (not isinstance(result, torch.Tensor) or result.ndim != 4
            or result.shape[:2] != (1, 128)
            or result.shape[-2] % h or result.shape[-1] % w
            or result.shape[-2] // h != result.shape[-1] // w
            or result.shape[-2] // h not in (2, 4)):
        raise ValueError("Steppe must return a 2x or 4x packed FLUX.3 image latent")
    height, width = result.shape[-2:]
    rows, cols = torch.meshgrid(torch.arange(height, device=ids.device),
                                torch.arange(width, device=ids.device), indexing="ij")
    zeros = torch.zeros_like(rows)
    output_ids = torch.stack((zeros, rows, cols, zeros), -1).reshape(-1, 4).to(ids)
    output_tokens = result[0].permute(1, 2, 0).reshape(-1, 128).to(tokens)
    return output_tokens, output_ids


@contextmanager
def decoder_scope(callback):
    """Cover original, experimental distilled and exact Base runtime dispatch."""
    runtime = importlib.import_module("flux3_forge.runtime")
    comfy = importlib.import_module("flux3_forge.comfy_bridge")
    ae_module = importlib.import_module("flux3_forge.vendor.flux3.autoencoder")
    decoder_type = ae_module.AutoEncoder
    with _lock:
        owner_thread = threading.get_ident()
        request_token = _request_callback.set(callback)
        original_decode = decoder_type.decode
        engines = [(module, module.generate) for module in (runtime, comfy)]

        @wraps(original_decode)
        def decode(ae, tokens, ids):
            context = _active.get()
            if context is not None:
                fn, seed = context
                bn = getattr(ae, "bn", None)
                if (bn is None or bn.running_mean.numel() != 128
                        or bn.running_var.numel() != 128):
                    raise ValueError("Steppe FLUX.3 requires the official 128-channel BN VAE")
                tokens, ids = upscale_tokens(tokens, ids, fn, seed)
            return original_decode(ae, tokens, ids)

        def wrap_engine(original):
            @wraps(original)
            def generate(request, *args, **kwargs):
                if threading.get_ident() != owner_thread or _request_callback.get() is not callback:
                    return original(request, *args, **kwargs)
                token = _active.set((callback, request["seed"]))
                try:
                    return original(request, *args, **kwargs)
                finally:
                    _active.reset(token)
            return generate

        # No context is active outside a matched engine invocation. Other
        # threads using the same decoder continue to call its original method.
        replacements = [(module, original, wrap_engine(original)) for module, original in engines]
        decoder_type.decode = decode
        try:
            for module, original, replacement in replacements:
                module.generate = replacement
            yield
        finally:
            _request_callback.reset(request_token)
            if decoder_type.decode is decode:
                decoder_type.decode = original_decode
            for module, original, replacement in replacements:
                if module.generate is replacement:
                    module.generate = original


def install(callback_factory):
    """Call at app_started (or retry later); False means FLUX.3 is absent.

    factory(processing) returns None when disabled, otherwise a function
    (packed_B128HW, actual_image_seed) -> upscaled_B128HW. Exceptions propagate
    through FLUX.3's normal error handling; never silently produce a wrong size.
    """
    global _installed, _installed_owner, _installed_original, _installation_state
    try:
        bridge = importlib.import_module("flux3_forge.bridge")
    except ModuleNotFoundError as error:
        if error.name in ("flux3_forge", "flux3_forge.bridge"):
            return False
        raise
    if _installed is not None and bridge.generate is _installed:
        return True
    original = bridge.generate
    state = {"enabled": True}

    @wraps(original)
    def generate(processing, *args, **kwargs):
        if not state["enabled"]:
            return original(processing, *args, **kwargs)
        callback = callback_factory(processing)
        if callback is None:
            return original(processing, *args, **kwargs)
        with decoder_scope(callback):
            return original(processing, *args, **kwargs)

    bridge.generate = _installed = generate
    _installed_owner, _installed_original = bridge, original
    _installation_state = state
    return True


def uninstall(*_):
    """Register with on_script_unloaded; leave later third-party hooks intact."""
    global _installed, _installed_owner, _installed_original, _installation_state
    if _installation_state is not None:
        _installation_state["enabled"] = False
    if _installed_owner is not None and _installed_owner.generate is _installed:
        _installed_owner.generate = _installed_original
    _installed = _installed_owner = _installed_original = None
    _installation_state = None
