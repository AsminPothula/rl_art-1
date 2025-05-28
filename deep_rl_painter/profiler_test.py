import torch
import torch.profiler

# Dummy model and input
model = torch.nn.Linear(10, 10).cuda()
x = torch.randn(32, 10).cuda()

# Profiler with TensorBoard handler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/profiler_test"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        y = model(x)
        loss = y.sum()
        loss.backward()
        prof.step()
