import torch
STREAMING_VAE = True
# COMPILE = True
import os
COMPILE = os.getenv("ENABLE_COMPILE", "true").lower() == "true"
COMPILE_MODE = os.getenv("LIVEAVATAR_COMPILE_MODE") or None
COMPILE_BACKEND = os.getenv("LIVEAVATAR_COMPILE_BACKEND", "inductor")
_compile_dynamic_env = os.getenv("LIVEAVATAR_COMPILE_DYNAMIC")
if _compile_dynamic_env is None:
    COMPILE_DYNAMIC = None
else:
    COMPILE_DYNAMIC = _compile_dynamic_env.lower() == "true"
print(f"COMPILE: {COMPILE}")
torch._dynamo.config.cache_size_limit = 128

NO_REFRESH_INFERENCE = False

def is_compile_supported():
    return hasattr(torch, "compiler") and hasattr(torch.nn.Module, "compile")

def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func

def conditional_compile(func):
    if COMPILE:
        return torch.compile(
            mode=COMPILE_MODE,
            backend=COMPILE_BACKEND,
            dynamic=COMPILE_DYNAMIC,
        )(func)
    else:
        return func
