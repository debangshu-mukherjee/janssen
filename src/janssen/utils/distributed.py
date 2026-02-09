"""Multi-device utilities for scalable optical computing.

Extended Summary
----------------
Provides utilities for distributed computing with JAX, enabling efficient
parallel processing across multiple devices (CPUs, GPUs, or TPUs). This
module simplifies device mesh creation and array sharding for data parallelism
in optical simulations.

Routine Listings
----------------
get_device_memory_gb : function
    Detects device memory in GB (nvidia-smi for GPUs, system RAM for CPUs).
create_mesh : function
    Creates a device mesh for data parallelism across available devices.
get_device_count : function
    Gets the number of available JAX devices.
shard_batch : function
    Shards array data across the batch dimension for parallel processing.

Notes
-----
This module is designed for scaling optical simulations across multiple
devices. The batch dimension is sharded by default, making it ideal for
processing multiple optical fields or wavefronts in parallel.

Examples
--------
>>> import jax.numpy as jnp
>>> from janssen.utils.distributed import create_mesh, shard_batch
>>>
>>> # Create a mesh using all available devices
>>> mesh = create_mesh()
>>>
>>> # Shard a batch of data across devices
>>> data = jnp.ones((8, 256, 256))
>>> sharded_data = shard_batch(data, mesh)
"""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Optional, Tuple
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Shaped, jaxtyped


def get_device_memory_gb() -> Tuple[int, float]:
    """Detect device count and memory per device.

    Attempts to detect the number of JAX devices and total memory per
    device using platform-specific methods:
    - NVIDIA GPUs: nvidia-smi
    - CPUs: system RAM via psutil or /proc/meminfo
    - Other platforms: conservative fallback

    Returns
    -------
    num_devices : int
        Number of JAX devices detected via jax.device_count()
    memory_per_device_gb : float
        Memory per device in GB. For CPUs, returns total system RAM.
        Falls back to 16.0 GB for GPUs or 8.0 GB for unknown platforms.

    Notes
    -----
    **Detection Methods by Platform**:

    **NVIDIA GPUs**:
    - Uses nvidia-smi to query first GPU's total memory
    - Converts from MB to GB
    - Multi-GPU systems: assumes all GPUs have same memory

    **CPUs**:
    - First tries psutil.virtual_memory() for cross-platform detection
    - Falls back to /proc/meminfo on Linux if psutil unavailable
    - Returns total system RAM (all devices share this pool)

    **TPUs/Other Accelerators**:
    - No direct memory detection available
    - Falls back to conservative defaults

    **Fallback Values**:

    GPU fallback (16.0 GB) works for:
    - NVIDIA V100 (16 GB variant)
    - NVIDIA Tesla T4 (16 GB)
    - NVIDIA RTX 6000 (24 GB, safe to use 16)
    - NVIDIA A100 (40 GB variant, safe to use 16)

    CPU fallback (8.0 GB):
    - Conservative for modern systems (most have 16+ GB)
    - Prevents OOM on low-memory machines

    **Platform Limitations**:

    - NVIDIA GPU: Only detects first GPU memory
    - AMD GPU (ROCm): No detection, uses fallback
    - Intel GPU: No detection, uses fallback
    - Google TPU: No detection, uses fallback
    - CPU: Detects total system RAM (shared across all cores)

    Examples
    --------
    >>> from janssen.utils.distributed import get_device_memory_gb
    >>> num_devices, memory = get_device_memory_gb()
    >>> print(f"Detected {num_devices} devices with {memory:.1f} GB each")
    Detected 8 devices with 16.0 GB each

    See Also
    --------
    get_device_count : Get number of available devices
    """
    import jax

    num_devices: int = jax.device_count()
    platform: str = jax.devices()[0].platform

    if platform == "gpu":
        try:
            import subprocess

            result: subprocess.CompletedProcess = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                memory_mb: float = float(result.stdout.strip().split("\n")[0])
                memory_gb: float = memory_mb / 1024.0
                return (num_devices, memory_gb)
        except Exception:
            pass
        return (num_devices, 16.0)

    elif platform == "cpu":
        try:
            import psutil

            total_ram_bytes: int = psutil.virtual_memory().total
            total_ram_gb: float = total_ram_bytes / 1e9
            return (num_devices, total_ram_gb)
        except ImportError:
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_kb: int = int(line.split()[1])
                            mem_gb: float = mem_kb / 1e6
                            return (num_devices, mem_gb)
            except Exception:
                pass
        return (num_devices, 8.0)

    else:
        return (num_devices, 8.0)


@jaxtyped(typechecker=beartype)
def get_device_count() -> int:
    """Get number of available JAX devices.

    Returns
    -------
    n_devices : int
        Number of available accelerators (GPUs/TPUs)

    Examples
    --------
    >>> import janssen as jns
    >>> n = jns.utils.get_device_count()
    >>> print(f"Found {n} devices")
    Found 8 devices
    """
    n_devices: int = jax.device_count()
    return n_devices


@jaxtyped(typechecker=beartype)
def create_mesh(n_devices: Optional[int] = None) -> Mesh:
    """Create a device mesh for data parallelism.

    Creates a 1D device mesh suitable for sharding arrays across their
    batch dimension. The mesh can be used with sharding specifications
    to distribute computation across multiple devices.

    Parameters
    ----------
    n_devices : int, optional
        Number of devices to use in the mesh. If None, uses all available
        devices detected by JAX (default: None).

    Returns
    -------
    mesh : Mesh
        Device mesh with axis name 'batch' for data parallelism. The mesh
        contains a linear arrangement of devices for distributing batched
        computations.

    Examples
    --------
    >>> # Create mesh with all available devices
    >>> mesh = create_mesh()
    >>> print(mesh.shape)
    {'batch': 4}  # If 4 devices are available

    >>> # Create mesh with specific number of devices
    >>> mesh = create_mesh(n_devices=2)
    >>> print(mesh.shape)
    {'batch': 2}

    Notes
    -----
    The returned mesh has a single axis named 'batch', making it suitable
    for distributing the first dimension of arrays across devices. For more
    complex sharding patterns, consider creating custom meshes using
    jax.sharding.Mesh directly.
    """
    if n_devices is None:
        n_devices = jax.device_count()
    all_devices: list = jax.devices()
    selected_devices: list = all_devices[:n_devices]
    devices: jnp.ndarray = mesh_utils.create_device_mesh(
        (n_devices,), devices=selected_devices
    )
    return Mesh(devices, axis_names=("batch",))


@jaxtyped(typechecker=beartype)
def shard_batch(
    data: Shaped[Array, " ..."], mesh: Mesh
) -> Shaped[Array, " ..."]:
    """Shard data across batch dimension.

    Distributes an array's first dimension (batch dimension) across devices
    in the provided mesh. This enables parallel processing of batched data
    with automatic memory distribution and computation parallelism.

    Parameters
    ----------
    data : Shaped[Array, " ..."]
        Input array to shard. The first dimension is treated as the batch
        dimension and will be distributed across devices. Can be any JAX
        or NumPy array.
    mesh : Mesh
        Device mesh created by create_mesh() or custom mesh with a 'batch'
        axis. Defines how the data will be distributed across devices.

    Returns
    -------
    sharded_data : Shaped[Array, " ..."]
        Input array with the batch dimension sharded across devices in the
        mesh. The array's computation will be automatically parallelized
        across devices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from janssen.utils.distributed import create_mesh, shard_batch
    >>>
    >>> # Create sample data with batch dimension
    >>> data = jnp.ones((8, 256, 256))
    >>>
    >>> # Create mesh and shard data
    >>> mesh = create_mesh()
    >>> sharded_data = shard_batch(data, mesh)
    >>>
    >>> # The first dimension is now distributed across devices
    >>> # Operations on sharded_data will run in parallel

    Notes
    -----
    - The batch dimension size should ideally be divisible by the number
      of devices for optimal load balancing.
    - Sharding is applied using NamedSharding with PartitionSpec('batch'),
      which partitions only the first dimension.
    - Subsequent operations on the sharded array will automatically
      maintain the sharding pattern where possible.
    """
    sharding: NamedSharding = NamedSharding(mesh, P("batch"))
    return jax.device_put(data, sharding)
