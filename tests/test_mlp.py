import pytest
import torch
import torch.nn as nn
from hypertrees.models.mlp import MLP, RPLayer


class TestRPLayerInitialization:
    """Test RPLayer (Random Projection Layer) initialization and functionality."""
    
    def test_default_initialization(self):
        """Test RPLayer initialization with standard parameters."""
        layer = RPLayer(in_dim=10, out_dim=5, seed=42)
        
        assert layer.weight.shape == (5, 10)
        assert layer.bias is None
        assert not layer.weight.requires_grad  # Should be fixed
        
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same weights."""
        layer1 = RPLayer(in_dim=10, out_dim=5, seed=42)
        layer2 = RPLayer(in_dim=10, out_dim=5, seed=42)
        
        # Weights should be identical
        assert torch.allclose(layer1.weight, layer2.weight)
        
    def test_different_seeds_produce_different_weights(self):
        """Test that different seeds produce different weights."""
        layer1 = RPLayer(in_dim=10, out_dim=5, seed=42)
        layer2 = RPLayer(in_dim=10, out_dim=5, seed=123)
        
        # Weights should be different
        assert not torch.allclose(layer1.weight, layer2.weight)
        
    def test_different_dimensions(self):
        """Test RPLayer with different input/output dimensions."""
        dimensions = [(5, 3), (20, 10), (100, 50), (1, 1)]
        
        for in_dim, out_dim in dimensions:
            layer = RPLayer(in_dim=in_dim, out_dim=out_dim, seed=42)
            assert layer.weight.shape == (out_dim, in_dim)
            
    def test_forward_pass(self):
        """Test forward pass through RPLayer."""
        layer = RPLayer(in_dim=10, out_dim=5, seed=42)
        
        # Create input tensor
        x = torch.randn(3, 10)  # Batch size 3, input dim 10
        
        # Forward pass
        output = layer(x)
        
        # Check output shape
        assert output.shape == (3, 5)
        
        # Check that it's equivalent to linear transformation
        expected_output = torch.nn.functional.linear(x, layer.weight, layer.bias)
        assert torch.allclose(output, expected_output)
        
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        layer = RPLayer(in_dim=10, out_dim=5, seed=42)
        
        batch_sizes = [1, 5, 16, 32]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 10)
            output = layer(x)
            assert output.shape == (batch_size, 5)
            
    def test_weight_buffer_registration(self):
        """Test that weights are properly registered as buffers."""
        layer = RPLayer(in_dim=10, out_dim=5, seed=42)
        
        # Check that weight is in buffers
        assert 'weight' in layer._buffers
        assert 'bias' in layer._buffers
        assert layer._buffers['bias'] is None
        
        # Check that weight is persistent
        state_dict = layer.state_dict()
        assert 'weight' in state_dict


class TestMLPInitialization:
    """Test MLP initialization and parameter validation."""
    
    def test_default_initialization_without_random_projection(self):
        """Test MLP initialization without random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        # Check that layers are properly configured
        layers = list(mlp.layers.children())
        assert len(layers) == 4  # Linear, ReLU, Linear, Dropout
        
        # Check layer types
        assert isinstance(layers[0], nn.Linear)  # Input layer
        assert isinstance(layers[1], nn.ReLU)    # Activation
        assert isinstance(layers[2], nn.Linear)  # Output layer
        assert isinstance(layers[3], nn.Dropout) # Dropout
        
        # Check dimensions
        assert layers[0].in_features == 10   # tree_embed_dim
        assert layers[0].out_features == 20  # hidden_dim
        assert layers[2].in_features == 20   # hidden_dim
        assert layers[2].out_features == 5   # output_dim
        
    def test_initialization_with_random_projection(self):
        """Test MLP initialization with random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=True,
            rp_embed_dim=8,
            seed=42
        )
        
        # Check that RP layer is included
        layers = list(mlp.layers.children())
        assert len(layers) == 5  # RP, Linear, ReLU, Linear, Dropout
        
        # Check layer types and order
        assert isinstance(layers[0], RPLayer)    # Random projection
        assert isinstance(layers[1], nn.Linear)  # Input layer
        assert isinstance(layers[2], nn.ReLU)    # Activation
        assert isinstance(layers[3], nn.Linear)  # Output layer
        assert isinstance(layers[4], nn.Dropout) # Dropout
        
        # Check RP layer dimensions
        assert layers[0].weight.shape == (8, 10)  # rp_embed_dim, tree_embed_dim
        
        # Check linear layer dimensions after RP
        assert layers[1].in_features == 8    # rp_embed_dim
        assert layers[1].out_features == 20  # hidden_dim
        
    def test_custom_parameters(self):
        """Test MLP initialization with custom parameters."""
        mlp = MLP(
            tree_embed_dim=15,
            output_dim=3,
            hidden_dim=50,
            use_random_projection=False,
            dropout_rate=0.3,
            seed=123
        )
        
        layers = list(mlp.layers.children())
        
        # Check dimensions
        assert layers[0].in_features == 15   # tree_embed_dim
        assert layers[0].out_features == 50  # hidden_dim
        assert layers[2].in_features == 50   # hidden_dim
        assert layers[2].out_features == 3   # output_dim
        
        # Check dropout rate
        dropout_layer = layers[3]
        assert isinstance(dropout_layer, nn.Dropout)
        assert dropout_layer.p == 0.3
        
    def test_missing_rp_embed_dim_with_random_projection(self):
        """Test that using random projection without rp_embed_dim raises ValueError."""
        with pytest.raises(ValueError, match="rp_embed_dim must be provided"):
            MLP(
                tree_embed_dim=10,
                output_dim=5,
                hidden_dim=20,
                use_random_projection=True,
                rp_embed_dim=None
            )


class TestMLPForwardPass:
    """Test MLP forward pass functionality."""
    
    def test_forward_pass_without_random_projection(self):
        """Test forward pass without random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=False
        )
        
        # Create input tensor
        x = torch.randn(3, 10)  # Batch size 3, input dim 10
        
        # Forward pass
        output = mlp(x)
        
        # Check output shape
        assert output.shape == (3, 5)
        
        # Check that output is finite
        assert torch.all(torch.isfinite(output))
        
    def test_forward_pass_with_random_projection(self):
        """Test forward pass with random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=True,
            rp_embed_dim=8,
            seed=42
        )
        
        # Create input tensor
        x = torch.randn(3, 10)  # Batch size 3, input dim 10
        
        # Forward pass
        output = mlp(x)
        
        # Check output shape
        assert output.shape == (3, 5)
        
        # Check that output is finite
        assert torch.all(torch.isfinite(output))
        
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        batch_sizes = [1, 5, 16, 32, 64]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 10)
            output = mlp(x)
            assert output.shape == (batch_size, 5)
            
    def test_rp_layer_reproducibility(self):
        """Test that the random projection layer is reproducible with the same seed."""
        mlp1 = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=True,
            rp_embed_dim=8,
            seed=42
        )

        mlp2 = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=True,
            rp_embed_dim=8,
            seed=42
        )

        rp1 = list(mlp1.layers.children())[0]
        rp2 = list(mlp2.layers.children())[0]
        assert torch.equal(rp1.weight, rp2.weight)
        
    def test_training_vs_eval_mode(self):
        """Test difference between training and evaluation modes (dropout)."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            dropout_rate=0.5  # High dropout to see difference
        )
        
        x = torch.randn(100, 10)  # Larger batch to see statistical difference
        
        # Training mode
        mlp.train()
        output_train = mlp(x)
        
        # Eval mode
        mlp.eval()
        output_eval1 = mlp(x)
        output_eval2 = mlp(x)
        
        # In eval mode, outputs should be identical
        assert torch.allclose(output_eval1, output_eval2)
        
        # Training mode might be different (due to dropout randomness)
        # But we can't guarantee this in a single test, so just check shapes
        assert output_train.shape == output_eval1.shape == (100, 5)


class TestMLPArchitectureVariations:
    """Test different MLP architecture configurations."""
    
    def test_different_dimensions(self):
        """Test MLP with various dimension combinations."""
        test_configs = [
            (5, 3, 10),      # Small network
            (100, 50, 200),  # Medium network
            (1, 1, 2),       # Minimal network
            (64, 128, 32),   # Common sizes
        ]
        
        for tree_dim, output_dim, hidden_dim in test_configs:
            mlp = MLP(
                tree_embed_dim=tree_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim
            )
            
            # Test forward pass
            x = torch.randn(2, tree_dim)
            output = mlp(x)
            assert output.shape == (2, output_dim)
            
    def test_different_dropout_rates(self):
        """Test MLP with different dropout rates."""
        dropout_rates = [0.0, 0.1, 0.3, 0.5, 0.8]
        
        for rate in dropout_rates:
            mlp = MLP(
                tree_embed_dim=10,
                output_dim=5,
                hidden_dim=20,
                dropout_rate=rate
            )
            
            # Check dropout layer
            layers = list(mlp.layers.children())
            dropout_layer = layers[-1]
            assert isinstance(dropout_layer, nn.Dropout)
            assert dropout_layer.p == rate
            
            # Test forward pass
            x = torch.randn(3, 10)
            output = mlp(x)
            assert output.shape == (3, 5)
            
    def test_random_projection_dimensions(self):
        """Test MLP with different random projection dimensions."""
        rp_dimensions = [2, 5, 8, 16, 32]
        
        for rp_dim in rp_dimensions:
            mlp = MLP(
                tree_embed_dim=20,
                output_dim=5,
                hidden_dim=15,
                use_random_projection=True,
                rp_embed_dim=rp_dim,
                seed=42
            )
            
            # Check RP layer dimensions
            layers = list(mlp.layers.children())
            rp_layer = layers[0]
            assert isinstance(rp_layer, RPLayer)
            assert rp_layer.weight.shape == (rp_dim, 20)
            
            # Check subsequent linear layer
            linear_layer = layers[1]
            assert linear_layer.in_features == rp_dim
            
            # Test forward pass
            x = torch.randn(3, 20)
            output = mlp(x)
            assert output.shape == (3, 5)


class TestMLPDeviceHandling:
    """Test MLP behavior with different devices."""
    
    def test_cpu_operations(self):
        """Test MLP operations on CPU."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        # Ensure on CPU
        mlp = mlp.to('cpu')
        
        x = torch.randn(3, 10)
        output = mlp(x)
        
        assert output.device.type == 'cpu'
        assert output.shape == (3, 5)
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_operations(self):
        """Test MLP operations on CUDA."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        # Move to CUDA
        mlp = mlp.to('cuda')
        
        x = torch.randn(3, 10).to('cuda')
        output = mlp(x)
        
        assert output.device.type == 'cuda'
        assert output.shape == (3, 5)
        
    def test_device_consistency_with_random_projection(self):
        """Test device consistency when using random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=True,
            rp_embed_dim=8,
            seed=42
        )
        
        # Check that all components are on the same device initially
        for param in mlp.parameters():
            assert param.device.type == 'cpu'
            
        for buffer in mlp.buffers():
            assert buffer.device.type == 'cpu'


class TestMLPGradientFlow:
    """Test gradient flow through MLP."""
    
    def test_gradients_without_random_projection(self):
        """Test that gradients flow properly without random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        x = torch.randn(3, 10, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check that model parameters have gradients
        for param in mlp.parameters():
            assert param.grad is not None
            
    def test_gradients_with_random_projection(self):
        """Test that gradients flow properly with random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=True,
            rp_embed_dim=8,
            seed=42
        )
        
        x = torch.randn(3, 10, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Check that trainable parameters have gradients
        # RP layer weights should NOT have gradients (they're fixed)
        rp_layer = mlp.layers[0]
        assert rp_layer.weight.grad is None  # Fixed weights
        
        # Linear layers should have gradients
        for layer in mlp.layers:
            if isinstance(layer, nn.Linear):
                assert layer.weight.grad is not None
                if layer.bias is not None:
                    assert layer.bias.grad is not None


class TestMLPEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_dropout_rate(self):
        """Test MLP with zero dropout rate."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            dropout_rate=0.0
        )
        
        x = torch.randn(3, 10)
        
        # Training and eval mode should give same results with 0 dropout
        mlp.train()
        output_train = mlp(x)
        
        mlp.eval()
        output_eval = mlp(x)
        
        assert torch.allclose(output_train, output_eval)
        
    def test_very_small_dimensions(self):
        """Test MLP with minimal dimensions."""
        mlp = MLP(
            tree_embed_dim=1,
            output_dim=1,
            hidden_dim=1
        )
        
        x = torch.randn(1, 1)
        output = mlp(x)
        
        assert output.shape == (1, 1)
        assert torch.all(torch.isfinite(output))
        
    def test_large_batch_size(self):
        """Test MLP with large batch size."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        x = torch.randn(1000, 10)
        output = mlp(x)
        
        assert output.shape == (1000, 5)
        assert torch.all(torch.isfinite(output))
        
    def test_random_projection_larger_than_input(self):
        """Test random projection with output dimension larger than input."""
        mlp = MLP(
            tree_embed_dim=5,
            output_dim=3,
            hidden_dim=10,
            use_random_projection=True,
            rp_embed_dim=8,  # Larger than tree_embed_dim=5
            seed=42
        )
        
        x = torch.randn(2, 5)
        output = mlp(x)
        
        assert output.shape == (2, 3)
        
        # Check RP layer dimensions
        rp_layer = mlp.layers[0]
        assert rp_layer.weight.shape == (8, 5)


class TestMLPStateDictAndSerialization:
    """Test MLP state dict and model serialization."""
    
    def test_state_dict_without_random_projection(self):
        """Test state dict for MLP without random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        state_dict = mlp.state_dict()
        
        # Should contain weights and biases for linear layers
        expected_keys = [
            'layers.0.weight', 'layers.0.bias',  # First linear layer
            'layers.2.weight', 'layers.2.bias'   # Second linear layer
        ]
        
        for key in expected_keys:
            assert key in state_dict
            
    def test_state_dict_with_random_projection(self):
        """Test state dict for MLP with random projection."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            use_random_projection=True,
            rp_embed_dim=8,
            seed=42
        )
        
        state_dict = mlp.state_dict()
        
        # Should contain RP weights and linear layer weights
        expected_keys = [
            'layers.0.weight',              # RP layer weight
            'layers.1.weight', 'layers.1.bias',  # First linear layer
            'layers.3.weight', 'layers.3.bias'   # Second linear layer
        ]
        
        for key in expected_keys:
            assert key in state_dict
            
        # RP layer should not have bias
        assert 'layers.0.bias' not in state_dict or state_dict['layers.0.bias'] is None
        
    def test_load_state_dict(self):
        """Test loading state dict."""
        mlp1 = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            seed=42
        )
        
        mlp2 = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20,
            seed=123  # Different seed
        )
        
        # Save state dict from first model
        state_dict = mlp1.state_dict()
        
        # Load into second model
        mlp2.load_state_dict(state_dict)
        
        # Both models should now give same output
        x = torch.randn(3, 10)
        
        mlp1.eval()
        mlp2.eval()
        
        output1 = mlp1(x)
        output2 = mlp2(x)
        
        assert torch.allclose(output1, output2)


class TestMLPIntegration:
    """Integration tests for MLP in realistic scenarios."""
    
    def test_training_loop_simulation(self):
        """Simulate a basic training loop."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=3,
            hidden_dim=20,
            dropout_rate=0.1
        )
        
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Generate synthetic data
        x = torch.randn(32, 10)
        y = torch.randn(32, 3)
        
        initial_loss = None
        final_loss = None
        
        # Training loop
        for epoch in range(10):
            mlp.train()
            optimizer.zero_grad()
            
            output = mlp(x)
            loss = criterion(output, y)
            
            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 9:
                final_loss = loss.item()
                
            loss.backward()
            optimizer.step()
            
        # Loss should generally decrease (though not guaranteed with random data)
        assert isinstance(final_loss, float)
        assert torch.isfinite(torch.tensor(final_loss))
        
    def test_with_different_optimizers(self):
        """Test MLP with different optimizers."""
        mlp = MLP(
            tree_embed_dim=10,
            output_dim=5,
            hidden_dim=20
        )
        
        optimizers = [
            torch.optim.SGD(mlp.parameters(), lr=0.01),
            torch.optim.Adam(mlp.parameters(), lr=0.001),
            torch.optim.RMSprop(mlp.parameters(), lr=0.01)
        ]
        
        x = torch.randn(5, 10)
        y = torch.randn(5, 5)
        criterion = nn.MSELoss()
        
        for optimizer in optimizers:
            # Reset model parameters
            mlp.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            
            # Single optimization step
            optimizer.zero_grad()
            output = mlp(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            # Check that optimization step completed without error
            assert torch.all(torch.isfinite(output))
            
    def test_ensemble_simulation(self):
        """Test using multiple MLP instances as an ensemble."""
        ensemble_size = 3
        mlps = []
        
        for i in range(ensemble_size):
            mlp = MLP(
                tree_embed_dim=10,
                output_dim=5,
                hidden_dim=20,
                use_random_projection=True,
                rp_embed_dim=8,
                seed=42 + i  # Different seeds for diversity
            )
            mlps.append(mlp)
            
        x = torch.randn(4, 10)
        
        # Get forecasts from all models
        predictions = []
        for mlp in mlps:
            mlp.eval()
            with torch.no_grad():
                pred = mlp(x)
                predictions.append(pred)
                
        # Ensemble forecast (mean)
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        
        assert ensemble_pred.shape == (4, 5)
        assert torch.all(torch.isfinite(ensemble_pred))
        
        # Individual forecasts should be different (due to different seeds)
        for i in range(1, len(predictions)):
            assert not torch.allclose(predictions[0], predictions[i])
