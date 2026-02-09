"""Multi-device utilities for scalable optical computing.

Extended Summary
----------------
Provides utilities for distributed computing with JAX, enabling efficient
parallel processing across multiple devices (CPUs, GPUs, or TPUs). This
module simplifies device mesh creation and array sharding for data parallelism
in optical simulations.

Routine Listings
----------------
get_gpu_memory_gb : function
    Detects GPU memory in GB using nvidia-smi or returns default.
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
from beartype.typing import Optional
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Shaped, jaxtyped


def get_gpu_memory_gb() -> float:
    """Detect GPU memory in GB using nvidia-smi.

    Attempts to detect the total memory of the first GPU using nvidia-smi.
    Falls back to 16.0 GB (a conservative default for V100/A100 GPUs) if
    detection fails.

    Returns
    -------
    memory_gb : float
        Total GPU memory in GB. Returns 16.0 if detection fails or if
        nvidia-smi is not available.

    Notes
    -----
    **Detection Method**:

    Uses nvidia-smi command-line tool to query GPU memory:
    - Queries the first GPU's total memory
    - Converts from MB to GB
    - Returns immediately on first successful query

    **Fallback Behavior**:

    Returns 16.0 GB if:
    - nvidia-smi is not installed or not in PATH
    - nvidia-smi query fails
    - Output parsing fails
    - Timeout occurs (5 second limit)
    - Non-NVIDIA GPUs are used (AMD, Intel, TPU)

    The 16.0 GB default is chosen as a conservative value that works for:
    - NVIDIA V100 (16 GB variant)
    - NVIDIA Tesla T4 (16 GB)
    - NVIDIA RTX 6000 (24 GB, safe to use 16)
    - NVIDIA A100 (40 GB variant, safe to use 16)

    **Platform Limitations**:

    - Only works for NVIDIA GPUs with nvidia-smi installed
    - Does not detect AMD GPU memory (ROCm)
    - Does not detect Intel GPU memory
    - Does not detect Google TPU memory
    - For multi-GPU systems, returns memory of first GPU only

    Examples
    --------
    >>> from janssen.utils.distributed import get_gpu_memory_gb
    >>> memory = get_gpu_memory_gb()
    >>> print(f"Detected {memory:.1f} GB GPU memory")
    Detected 16.0 GB GPU memory

    See Also
    --------
    get_device_count : Get number of available devices
    """
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
            return memory_gb
    except Exception:
        pass
    return 16.0


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
