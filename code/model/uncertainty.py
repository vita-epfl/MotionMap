import os
import torch
from matplotlib import pyplot as plt

# Define a list of color pairs for plotting, used to differentiate joints visually
# Reyhaneh really loves colors 😭
color_pairs = [
        ("lightseagreen", "turquoise"),            # 0 ("palevioletred" if j<l_j else "pink")
        ("palevioletred" , "pink"),                # 1
        ("lightskyblue", "steelblue"), 
        ("thistle", "mediumorchid"),                   
        ("khaki", "gold"),                         # 5
        ("lightgreen", "limegreen"),               # 6
        ("sandybrown", "chocolate"),               # 7
        ("lightpink", "palevioletred"),            # 8
        ("powderblue", "lightslategray"),          # 9
        ("peachpuff", "darkorange"),               # 10
        ("lavender", "rebeccapurple"),             # 11 - Added purplish
        ("plum", "purple"),                        # 12 - Purplish
        ("violet", "darkviolet"),                  # 13 - Added purplish
        ("orchid", "darkorchid"),                  # 14 - Purplish
        ("palegoldenrod", "goldenrod"),            # 15
        ("azure", "navy"),                         # 16 - Contrast with a deep color
        ("mistyrose", "firebrick"),                # 17
        ("lemonchiffon", "khaki"),                 # 18
        ("lightcyan", "cyan"),                     # 19
        ("lavender", "blueviolet")                 # 20 - Added purplish
    ]

class UncertaintyNetwork(torch.nn.Module):
    """
    A PyTorch module for modeling uncertainty in human pose forecasting.
    
    Attributes:
        num_joints (int): Number of joints in the human pose.
        save_path (str): Path to save uncertainty plots.
        total_length (int): Total number of frames (context + prediction).
        mlp (torch.nn.Sequential): A multi-layer perceptron for uncertainty estimation.
        epoch (int): Tracks the last epoch for plotting.
    """
    def __init__(self, arch, save_path):
        """
        Initializes the UncertaintyNetwork.

        Args:
            arch (dict): Architecture configuration containing 'n_landmarks', 
                         'pred_length', 'context_length', and 'hidden_dim'.
            save_path (str): Directory path to save plots.
        """
        super(UncertaintyNetwork, self).__init__()

        self.num_joints = arch['n_landmarks']
        self.save_path = save_path

        # Number of frames to predict
        total_length = arch['pred_length'] + arch['context_length']
        self.total_length = total_length
        
        output_size = total_length * self.num_joints
        
        # Used for plotting only
        self.epoch = -1

        latent_dim = 2 * arch['hidden_dim']

        self.mlp = [torch.nn.Linear(latent_dim, latent_dim) if i % 2 == 0 \
            else torch.nn.ELU() for i in range(2 * 5)]
        self.mlp.append(torch.nn.Linear(latent_dim, output_size))
        self.mlp = torch.nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)
    
    def loss(self, x, y, y_hat, epoch):
        """
        Computes the negative log-likelihood.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Ground truth tensor.
            y_hat (torch.Tensor): Predicted tensor.
            epoch (int): Current epoch number.

        Returns:
            tuple: Total loss and detached uncertainty values.
        """
        # Plot uncertainty every 10 epochs
        if epoch != self.epoch:
            if epoch % 10 == 0:
                self.plot_uncertainty(x, epoch)
            self.epoch = epoch

        # Compute log_sigma (uncertainty) predictions
        log_sigma = self(x)
        log_sigma = log_sigma.view(x.shape[0], self.total_length, self.num_joints)

        # Compute squared L2 loss
        loss = torch.norm(y_hat - y, dim=-1, p=2) ** 2
        loss = loss.squeeze(2)

        # Compute uncertainty-aware loss
        loss_detached = loss.clone().detach()
        uncertainty_loss = (loss_detached / torch.exp(log_sigma)) + log_sigma
        return loss + uncertainty_loss, torch.exp(log_sigma.detach())

    @torch.no_grad()
    def plot_uncertainty(self, x, epoch):
        """
        Plots the uncertainty for each joint and saves the plots.

        Args:
            x (torch.Tensor): Input tensor.
            epoch (int): Current epoch number.
        """
        # Compute log_sigma and convert to standard deviation (sigma)
        log_sigma = self(x)
        log_sigma = log_sigma.view(x.shape[0], self.total_length, self.num_joints)
        sigma = torch.exp(log_sigma)

        # Create a 4x4 grid of subplots
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.tight_layout()

        # Convert sigma to numpy for plotting
        sigmas = sigma.cpu().numpy()

        # Iterate over the batch
        for s, sigma in enumerate(sigmas):
            # Iterate over the joints
            for i in range(4): 
                for j in range(4):
                    # Plot uncertainty for each joint
                    axes[i, j].plot(sigma[:, i*4+j], ".", markersize=3, color=color_pairs[(s+2)%20][0])
                    axes[i, j].set_title("joint {}".format(i*4+j))
            
            # Add a legend for joint IDs
            axes[0, 0].legend(["id_{}".format(i) for i in range(sigma.shape[1])])

        # Create the directory if it doesn't exist
        if not os.path.exists(os.path.join(self.save_path, 'plots', 'uncertainty')):
            os.makedirs(os.path.join(self.save_path, 'plots', 'uncertainty'))

        # Save the plots as PNG and SVG
        fig.savefig(
            os.path.join(self.save_path, 'plots', 'uncertainty', "{}.png".format(epoch)),
            dpi=300, bbox_inches='tight'
        )
        fig.savefig(
            os.path.join(self.save_path, 'plots', 'uncertainty', "{}.svg".format(epoch)),
            bbox_inches='tight'
        )

        print("Saved uncertainty plots")
        plt.close(fig)