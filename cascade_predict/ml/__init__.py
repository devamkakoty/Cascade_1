try:
    from .pinn import CascadePINN
    from .dataset import CascadeDataset, generate_training_data
    from .train import Trainer
except ImportError:
    CascadePINN = None
    CascadeDataset = None
    generate_training_data = None
    Trainer = None
