"""Tests for distributed computing utilities in janssen.utils.distributed module."""

import chex
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from janssen.utils.distributed import create_mesh, get_device_count, shard_batch


class TestGetDeviceCount(chex.TestCase):
    """Test the get_device_count function."""

    def test_get_device_count_returns_int(self) -> None:
        """Test that get_device_count returns an integer."""
        n_devices = get_device_count()
        assert isinstance(n_devices, int)

    def test_get_device_count_positive(self) -> None:
        """Test that get_device_count returns a positive number."""
        n_devices = get_device_count()
        assert n_devices > 0

    def test_get_device_count_matches_jax(self) -> None:
        """Test that get_device_count matches jax.device_count()."""
        n_devices = get_device_count()
        assert n_devices == jax.device_count()

    def test_get_device_count_consistent(self) -> None:
        """Test that get_device_count returns consistent values."""
        n_devices_1 = get_device_count()
        n_devices_2 = get_device_count()
        assert n_devices_1 == n_devices_2


class TestCreateMesh(chex.TestCase):
    """Test the create_mesh function for device mesh creation."""

    def test_create_mesh_default(self) -> None:
        """Test creating a mesh with default settings (all available devices)."""
        mesh = create_mesh()

        # Check that mesh is created
        assert isinstance(mesh, Mesh)

        # Check that mesh has correct axis names
        assert mesh.axis_names == ("batch",)

        # Check that mesh uses all available devices
        assert mesh.shape["batch"] == jax.device_count()

        # Check that the mesh contains the correct number of devices
        assert len(mesh.devices.flat) == jax.device_count()

    def test_create_mesh_single_device(self) -> None:
        """Test creating a mesh with a single device."""
        mesh = create_mesh(n_devices=1)

        # Check that mesh is created
        assert isinstance(mesh, Mesh)

        # Check that mesh has correct shape
        assert mesh.shape["batch"] == 1

        # Check that mesh contains exactly one device
        assert len(mesh.devices.flat) == 1

    def test_create_mesh_multiple_devices(self) -> None:
        """Test creating a mesh with multiple devices if available."""
        n_available = jax.device_count()

        # Test with minimum of 2 devices or all available
        n_devices = min(2, n_available)
        mesh = create_mesh(n_devices=n_devices)

        # Check that mesh is created
        assert isinstance(mesh, Mesh)

        # Check that mesh has correct shape
        assert mesh.shape["batch"] == n_devices

        # Check that mesh contains correct number of devices
        assert len(mesh.devices.flat) == n_devices

    def test_create_mesh_axis_names(self) -> None:
        """Test that the mesh has the correct axis names."""
        mesh = create_mesh()

        # Check axis names tuple
        assert isinstance(mesh.axis_names, tuple)
        assert len(mesh.axis_names) == 1
        assert mesh.axis_names[0] == "batch"

    def test_create_mesh_idempotent(self) -> None:
        """Test that calling create_mesh multiple times works correctly."""
        mesh1 = create_mesh(n_devices=1)
        mesh2 = create_mesh(n_devices=1)

        # Both meshes should have the same properties
        assert mesh1.shape == mesh2.shape
        assert mesh1.axis_names == mesh2.axis_names


class TestShardBatch(chex.TestCase):
    """Test the shard_batch function for array sharding."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        super().setUp()
        self.mesh = create_mesh()

    def test_shard_batch_1d_array(self) -> None:
        """Test sharding a 1D array across the batch dimension."""
        # Create a 1D array
        n_devices = jax.device_count()
        data = jnp.arange(n_devices * 4)

        # Shard the data
        sharded_data = shard_batch(data, self.mesh)

        # Check that data is preserved
        chex.assert_trees_all_close(sharded_data, data)

        # Check that sharding is applied
        assert hasattr(sharded_data, "sharding")
        assert isinstance(sharded_data.sharding, NamedSharding)

    def test_shard_batch_2d_array(self) -> None:
        """Test sharding a 2D array across the batch dimension."""
        # Create a 2D array
        n_devices = jax.device_count()
        data = jnp.ones((n_devices * 8, 64))

        # Shard the data
        sharded_data = shard_batch(data, self.mesh)

        # Check that data is preserved
        chex.assert_trees_all_close(sharded_data, data)

        # Check shape is preserved
        assert sharded_data.shape == data.shape

    def test_shard_batch_3d_array(self) -> None:
        """Test sharding a 3D array (typical for optical fields)."""
        # Create a 3D array (batch, height, width)
        n_devices = jax.device_count()
        data = jnp.ones((n_devices * 4, 128, 128))

        # Shard the data
        sharded_data = shard_batch(data, self.mesh)

        # Check that data is preserved
        chex.assert_trees_all_close(sharded_data, data)

        # Check shape is preserved
        assert sharded_data.shape == data.shape

    def test_shard_batch_complex_array(self) -> None:
        """Test sharding complex arrays (optical fields)."""
        # Create a complex array
        n_devices = jax.device_count()
        data = jnp.ones((n_devices * 4, 64, 64), dtype=complex) * (1 + 1j)

        # Shard the data
        sharded_data = shard_batch(data, self.mesh)

        # Check that data is preserved
        chex.assert_trees_all_close(sharded_data, data)

        # Check dtype is preserved
        assert sharded_data.dtype == data.dtype

    def test_shard_batch_partition_spec(self) -> None:
        """Test that the correct partition spec is used."""
        # Create test data
        n_devices = jax.device_count()
        data = jnp.ones((n_devices * 4, 64))

        # Shard the data
        sharded_data = shard_batch(data, self.mesh)

        # Check that sharding uses correct partition spec
        expected_spec = P("batch")
        assert sharded_data.sharding.spec == expected_spec

    def test_shard_batch_preserves_operations(self) -> None:
        """Test that operations on sharded arrays work correctly."""
        # Create test data
        n_devices = jax.device_count()
        data = jnp.arange(n_devices * 8, dtype=float).reshape(-1, 1)

        # Shard the data
        sharded_data = shard_batch(data, self.mesh)

        # Perform operations
        result = sharded_data * 2 + 1

        # Check result is correct
        expected = data * 2 + 1
        chex.assert_trees_all_close(result, expected)

    def test_shard_batch_different_dtypes(self) -> None:
        """Test sharding arrays with different data types."""
        n_devices = jax.device_count()

        for dtype in [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]:
            data = jnp.ones((n_devices * 4, 32), dtype=dtype)
            sharded_data = shard_batch(data, self.mesh)

            # Check dtype is preserved
            assert sharded_data.dtype == dtype

            # Check data is preserved
            chex.assert_trees_all_close(sharded_data, data)

    def test_shard_batch_with_custom_mesh(self) -> None:
        """Test sharding with a custom mesh configuration."""
        # Create a custom mesh with specific device count
        custom_mesh = create_mesh(n_devices=1)

        # Create test data
        data = jnp.ones((4, 64, 64))

        # Shard the data
        sharded_data = shard_batch(data, custom_mesh)

        # Check that data is preserved
        chex.assert_trees_all_close(sharded_data, data)

    def test_shard_batch_empty_batch(self) -> None:
        """Test sharding an array with empty batch dimension."""
        # Create an array with zero batch size
        data = jnp.ones((0, 64, 64))

        # Shard the data
        sharded_data = shard_batch(data, self.mesh)

        # Check shape is preserved
        assert sharded_data.shape == data.shape

    def test_shard_batch_single_element_batch(self) -> None:
        """Test sharding an array with single element in batch."""
        # Create a mesh with single device to avoid sharding issues
        single_device_mesh = create_mesh(n_devices=1)

        # Create an array with batch size 1
        data = jnp.ones((1, 64, 64))

        # Shard the data using single device mesh
        sharded_data = shard_batch(data, single_device_mesh)

        # Check that data is preserved
        chex.assert_trees_all_close(sharded_data, data)


class TestDistributedIntegration(chex.TestCase):
    """Integration tests for distributed computing workflow."""

    def test_create_mesh_and_shard_workflow(self) -> None:
        """Test the complete workflow of creating mesh and sharding data."""
        # Step 1: Create mesh
        mesh = create_mesh()

        # Step 2: Create optical field data
        n_devices = jax.device_count()
        batch_size = n_devices * 4
        field_data = jnp.ones((batch_size, 256, 256), dtype=complex)

        # Step 3: Shard the data
        sharded_field = shard_batch(field_data, mesh)

        # Step 4: Perform computation
        result = jnp.abs(sharded_field) ** 2

        # Step 5: Verify result
        expected = jnp.ones((batch_size, 256, 256))
        chex.assert_trees_all_close(result, expected)

    def test_multiple_arrays_same_mesh(self) -> None:
        """Test sharding multiple arrays with the same mesh."""
        mesh = create_mesh()
        n_devices = jax.device_count()

        # Create multiple arrays
        array1 = jnp.ones((n_devices * 4, 64, 64))
        array2 = jnp.ones((n_devices * 4, 128, 128))
        array3 = jnp.ones((n_devices * 4, 32, 32))

        # Shard all arrays
        sharded1 = shard_batch(array1, mesh)
        sharded2 = shard_batch(array2, mesh)
        sharded3 = shard_batch(array3, mesh)

        # All should have same sharding on first dimension
        assert sharded1.sharding.spec == P("batch")
        assert sharded2.sharding.spec == P("batch")
        assert sharded3.sharding.spec == P("batch")

    def test_shard_transform_unshard(self) -> None:
        """Test sharding data, transforming it, and collecting results."""
        mesh = create_mesh()
        n_devices = jax.device_count()

        # Create input data
        input_data = jnp.arange(n_devices * 8, dtype=float).reshape(-1, 1)

        # Shard the data
        sharded_input = shard_batch(input_data, mesh)

        # Apply transformation
        transformed = sharded_input * 2 + 10

        # Collect result (implicitly by comparing with expected)
        expected = input_data * 2 + 10
        chex.assert_trees_all_close(transformed, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
