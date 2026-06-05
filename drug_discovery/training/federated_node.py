"""
Federated Learning Client Node

Implements the client-side logic for training and communicating with
the federated server.
"""

import logging
from collections import OrderedDict

import flwr as fl
import torch

logger = logging.getLogger(__name__)


class FederatedClient(fl.client.NumPyClient):
    """Flower client for federated training."""

    def __init__(self, model: torch.nn.Module, train_loader, test_loader, device: str = "cpu"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.model.to(self.device)

    def get_parameters(self, config: dict[str, any]) -> list[torch.Tensor]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list[torch.Tensor]):
        params_dict = zip(self.model.state_dict().keys(), parameters, strict=False)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list[torch.Tensor], config: dict[str, any]):
        self.set_parameters(parameters)

        # Local training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for _ in range(config.get("local_epochs", 1)):
            for batch in self.train_loader:
                optimizer.zero_grad()
                # Assuming batch is (data, target)
                data, target = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(data)
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters: list[torch.Tensor], config: dict[str, any]):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch in self.test_loader:
                data, target = batch[0].to(self.device), batch[1].to(self.device)
                output = self.model(data)
                loss += torch.nn.functional.mse_loss(output, target, reduction="sum").item()
                # Metrics logic here...

        return loss, len(self.test_loader.dataset), {"accuracy": 0.0}  # Placeholder accuracy


def start_client(server_address: str, client: FederatedClient):
    """Start the Flower client."""
    logger.info(f"Starting Federated Learning client connecting to {server_address}")
    fl.client.start_numpy_client(server_address=server_address, client=client)
