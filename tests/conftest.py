"""Keep CPU tests small and independent of the host's thread count."""
import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
